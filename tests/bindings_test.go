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

// Package bindings_test tests the generation of bindings for Gemma.
package bindings_test

import (
	"strings"
	"testing"

	"github.com/gx-org/gx/golang/binder"
	gxtesting "github.com/gx-org/gx/tests/testing"

	_ "github.com/gx-org/gemma/gemma"
)

func TestGemmaBindings(t *testing.T) {
	bld := gxtesting.NewBuilderStaticSource(nil)
	out := &strings.Builder{}
	pkg, err := bld.Build("github.com/gx-org/gemma/gemma")
	if err != nil {
		t.Fatalf("cannot generate bindings:\n%+v", err)
	}
	if err := binder.GoBindings(out, pkg.IR()); err != nil {
		t.Fatalf("cannot generate bindings:\n%+v", err)
	}
	got := out.String()
	for _, want := range []string{
		"SampleStep",
		"NewSamplingState",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("%q cannot be found in generated bindings:\n%s", want, got)
		}
	}
}
