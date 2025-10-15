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

// Utility promptcli prompts Gemma from the command line.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/gx-org/xlapjrt/plugin"
	"github.com/gx-org/gemma"
)

var (
	tokenizerModel = flag.String("tokenizer_model", "", "SentencePiece model to use for the tokenizer")
	inferenceModel = flag.String("inference_model", "", "Gemma weights in the GGUF format")
	prompt         = flag.String("prompt", "list me ten biggest cities", "Prompt to give to Gemma")
	pjrtPlugin     = flag.String("pjrt_plugin", "cpu", "PJRT plugin to load")
	model          = flag.String("model", "gemma-2b", "Model architecture to execute (must be gemma-2b or gemma3-270m)")
)

type GemmaPrompter interface {
	Prompt(prompt string) (string, error)
}

func main() {
	rtm, err := plugin.New(*pjrtPlugin)
	flag.Parse()
	if err != nil {
		fmt.Fprintf(os.Stderr, "%+v", err)
		os.Exit(1)
	}
	device, err := rtm.Device(0)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%+v", err)
		os.Exit(1)
	}
	var gem GemmaPrompter
	if *model == "gemma-2b" {
		gem, err = gemma.New(device, gemma.Params{
			NumSamplingSteps: 100,
			TokenizerModel:   *tokenizerModel,
			InferenceModel:   *inferenceModel,
		})
	} else if *model == "gemma3-270m" {
		gem, err = gemma.NewGemma3(device, gemma.Params{
			NumSamplingSteps: 512,
			TokenizerModel:   *tokenizerModel,
			InferenceModel:   *inferenceModel,
		})
	} else {
		fmt.Fprintf(os.Stderr, "Model type %q unsupported.\n", *model)
		os.Exit(1)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "%+v", err)
		os.Exit(1)
	}
	if _, err := gem.Prompt(*prompt); err != nil {
		fmt.Fprintf(os.Stderr, "%+v", err)
		os.Exit(1)
	}
}
