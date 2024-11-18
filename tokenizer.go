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

package gemma

import (
	"log"

	"github.com/pkg/errors"
	"github.com/eliben/go-sentencepiece"
	"github.com/gx-org/gx/golang/binder/gobindings/types"
)

// Tokenizer returns a new sentencepiece tokenizer.
type Tokenizer struct {
	gemma *Gemma
	sp    *sentencepiece.Processor
}

func (g *Gemma) newTokenizer() (*Tokenizer, error) {
	tk := &Tokenizer{gemma: g}
	var err error
	log.Printf("Reading tokenizer file %s", g.params.TokenizerModel)
	if tk.sp, err = sentencepiece.NewProcessorFromPath(g.params.TokenizerModel); err != nil {
		return nil, errors.Errorf("cannot load sentence piece: %v", err)
	}
	return tk, nil
}

// Encode a piece of text.
func (tk *Tokenizer) Encode(text string) (int, *types.HostArray[int32], error) {
	pieces := tk.sp.Encode(text)
	encoding := make([]int32, len(pieces)+1)
	encoding[0] = beginSentenceID
	for i, piece := range pieces {
		encoding[i+1] = int32(piece.ID)
	}
	return len(encoding), types.ArrayInt32(encoding), nil
}

// Decode an array of token IDs to text.
func (tk *Tokenizer) Decode(tokens []int) (string, error) {
	return tk.sp.Decode(tokens), nil
}
