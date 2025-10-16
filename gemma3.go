// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package gemma

import (
	"fmt"
	"io"
	"log"
	"os"
	"strings"

	ggufreader "github.com/abrander/gguf"
	"github.com/gx-org/gemma/gxdeps/github.com/gx-org/gemma/gemma/gemma_go_gx"
	"github.com/gx-org/gguf/encoding/gguf"
	"github.com/gx-org/gx/api"
	"github.com/gx-org/gx/golang/binder/gobindings/types"
)

type (
	// Gemma3 language model.
	Gemma3 struct {
		device    *api.Device
		params    Params
		tokenizer *Tokenizer
		gemmaGX   *gemma_go_gx.PackageHandle
		network   *gemma_go_gx.Gemma3
	}
)

// New Gemma language model instance.
func NewGemma3(device *api.Device, params Params) (g *Gemma3, err error) {
	g = &Gemma3{device: device, params: params}
	if g.tokenizer, err = newTokenizer(params); err != nil {
		return nil, fmt.Errorf("cannot create tokenizer: %v", err)
	}
	log.Printf("Reading model weights file %s", g.params.InferenceModel)
	fileReader, err := os.Open(g.params.InferenceModel)
	if err != nil {
		return nil, err
	}
	reader, err := ggufreader.Open(fileReader)
	if err != nil {
		return nil, err
	}
	vocabSize, err := vocabSize(reader)
	if err != nil {
		return nil, err
	}
	g.gemmaGX, err = gemma_go_gx.BuildHandleFor(device,
		gemma_go_gx.NumSamplingSteps.Set(int64(g.params.NumSamplingSteps)),
		gemma_go_gx.VocabSize.Set(int64(vocabSize)),
		gemma_go_gx.NumGemmaLayers.Set(18),
		gemma_go_gx.ModelDim.Set(640),
		gemma_go_gx.QKVDim.Set(256),
		gemma_go_gx.NumHeads.Set(4),
		gemma_go_gx.FFHiddenDim.Set(2*2048),
	)

	if err != nil {
		return nil, err
	}
	g.network = g.gemmaGX.Factory.NewGemma3()
	if err := gguf.UnmarshalOnDevice(device, g.network, gguf.ToReaders(reader)); err != nil {
		return nil, err
	}
	fmt.Println("Gemma3 initialized.")
	return
}

// Wrap returns the given user prompt with control tokens inserted.
//
// This assumes an instruction-tuned model.
func (g *Gemma3) Wrap(prompt string) string {
	return fmt.Sprintf("<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n", prompt)
}

func (g *Gemma3) Prompt(prompt string) (string, error) {
	log.Printf("Prompting Gemma3 with %q", prompt)
	promptSize, promptEncoded, err := g.tokenizer.Encode(g.Wrap(prompt))
	if err != nil {
		return "", fmt.Errorf("cannot encode prompt: %w", err)
	}
	g.gemmaGX.AppendOptions(gemma_go_gx.PromptLength.Set(int64(promptSize)))
	// NewSamplingState processes the prompt and returns the first predicted token.
	state, _, tokenHandle, err := g.network.NewSamplingState(promptEncoded)
	if err != nil {
		return "", fmt.Errorf("cannot create initial sampling state: %w", err)
	}

	output := []string{}
	stream := func(tokenHandle types.Atom[int64]) error {
		tokenID, err := tokenHandle.FetchValue()
		if err != nil {
			return err
		}
		if tokenID == 2 || tokenID == 106 {
			return io.EOF // EOS/EOT tokens, respectively.
		}
		token, err := g.tokenizer.Decode([]int{int(tokenID)})
		if err != nil {
			return err
		}
		fmt.Print(token)
		output = append(output, token)
		return nil
	}

	// Print the prompt, then the first token.
	fmt.Println(prompt)
	if err := stream(tokenHandle); err != nil {
		return "", err
	}

	for i := promptSize; i < g.params.NumSamplingSteps; i++ {
		newState, _, tokenHandle, err := g.network.SampleStep(state)
		if err != nil {
			return "", fmt.Errorf("cannot run sampling step %d: %w", i, err)
		}
		if err := stream(tokenHandle); err != nil {
			if err == io.EOF {
				break
			}
			return "", err
		}
		state = newState
	}

	return strings.Join(output, ""), nil
}
