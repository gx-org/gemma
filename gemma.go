// Copyright 2024 Google LLC
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

// Package gemma implements the gemma model.
package gemma

//go:generate go run github.com/gx-org/gx/golang/packager@main --gx_package=github.com/gx-org/gemma/gemma
//go:generate go run github.com/gx-org/gx/golang/binder/genbind@main --gx_package=github.com/gx-org/gemma/gemma

import (
	"fmt"
	"log"
	"os"
	"reflect"
	"strings"

	"github.com/gx-org/gemma/gxdeps/github.com/gx-org/gemma/gemma/gemma_go_gx"

	ggufreader "github.com/abrander/gguf"
	"github.com/gx-org/gguf/encoding/gguf"
	"github.com/gx-org/gx/api"
	"github.com/pkg/errors"
)

const (
	totalSamplingSteps = 100
	beginSentenceID    = 2
)

type (
	// Params are the parameters of Gemma and its components.
	Params struct {
		NumSamplingSteps int
		TokenizerModel   string
		InferenceModel   string
	}

	// Gemma language model.
	Gemma struct {
		device    *api.Device
		params    Params
		tokenizer *Tokenizer
		gemmaGX   *gemma_go_gx.PackageHandle
		network   *gemma_go_gx.Gemma
	}
)

func vocabSize(r *ggufreader.Reader) (int, error) {
	tokens, err := r.Metadata.Any("tokenizer.ggml.tokens")
	if err != nil {
		return 0, err
	}
	tokensS, ok := tokens.([]string)
	if !ok {
		return 0, errors.Errorf("cannot cast %T to %s", tokens, reflect.TypeFor[[]string]().String())
	}
	return len(tokensS), nil
}

// New Gemma language model instance.
func New(device *api.Device, params Params) (g *Gemma, err error) {
	g = &Gemma{device: device, params: params}
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
		gemma_go_gx.ModelDim.Set(2048),
		gemma_go_gx.QKVDim.Set(256),
		gemma_go_gx.NumHeads.Set(8),
		gemma_go_gx.FFHiddenDim.Set(8*2048),
	)
	if err != nil {
		return nil, err
	}
	g.network = g.gemmaGX.Factory.NewGemma()
	if err := gguf.UnmarshalOnDevice(device, g.network, gguf.ToReaders(reader)); err != nil {
		return nil, err
	}
	fmt.Println("Gemma initialized.")
	return
}

// Prompt Gemma with some text. Returns an answer.
func (g *Gemma) Prompt(prompt string) (string, error) {
	log.Printf("Prompting Gemma with %q", prompt)
	promptSize, promptEncoded, err := g.tokenizer.Encode(prompt)
	if err != nil {
		return "", fmt.Errorf("cannot encode prompt: %w", err)
	}
	g.gemmaGX.AppendOptions(gemma_go_gx.PromptLength.Set(int64(promptSize)))
	// NewSamplingState processes the prompt and returns the first predicted token.
	state, _, tokenHandle, err := g.network.NewSamplingState(promptEncoded)
	if err != nil {
		return "", fmt.Errorf("cannot create initial sampling state: %w", err)
	}

	// Print the first token.
	tokenID, err := tokenHandle.FetchValue()
	if err != nil {
		return "", err
	}
	if tokenID == 2 {
		return "", nil // EOS token.
	}
	token, err := g.tokenizer.Decode([]int{int(tokenID)})
	if err != nil {
		return "", err
	}
	fmt.Print(prompt, token)
	output := []string{token}

	for i := promptSize; i < g.params.NumSamplingSteps; i++ {
		newState, _, tokenHandle, err := g.network.SampleStep(state)
		if err != nil {
			return "", fmt.Errorf("cannot run sampling step %d: %w", i, err)
		}
		tokenID, err = tokenHandle.FetchValue()
		if err != nil {
			return "", err
		}
		if tokenID == 2 {
			break // EOS token.
		}
		token, err := g.tokenizer.Decode([]int{int(tokenID)})
		if err != nil {
			return "", err
		}
		fmt.Print(token)
		output = append(output, token)
		state = newState
	}

	return strings.Join(output, ""), nil
}
