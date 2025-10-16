// Package gemma encapsulates GX source files
// into a Go package.
//
// Automatically generated from google3/third_party/gxlang/gx/golang/packager/package.go.
//
// DO NOT EDIT
package gemma

import (
	"embed"

	"github.com/gx-org/gx/build/builder"
	"github.com/gx-org/gx/build/importers/embedpkg"

)

//go:embed gemma.gx gemma3.gx 
var srcs embed.FS

var inputFiles = []string{
"gemma.gx","gemma3.gx",
}

func init() {
	embedpkg.RegisterPackage("github.com/gx-org/gemma/gemma", Build)
}

var _ embedpkg.BuildFunc = Build

// Build GX package.
func Build(bld *builder.Builder) (builder.Package, error) {
	return bld.BuildFiles("github.com/gx-org/gemma", "gemma", srcs, inputFiles)
}
