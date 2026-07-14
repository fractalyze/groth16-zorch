// Copyright 2026 The Groth16Zorch Authors.
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

// gen_fixture creates a tiny gnark Groth16 fixture for testing the Python
// gnark loader.  Circuit: x × x == y  (1 constraint, a few wires).
//
// Usage:
//
//	go run . -output_dir=../data/tiny_multiply
package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"math/big"
	"os"
	"path/filepath"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark/backend/groth16"
	groth16bn254 "github.com/consensys/gnark/backend/groth16/bn254"
	"github.com/consensys/gnark/constraint"
	cs_bn254 "github.com/consensys/gnark/constraint/bn254"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
)

// TinyMultiply: x × x == y
type TinyMultiply struct {
	X frontend.Variable `gnark:",secret"`
	Y frontend.Variable `gnark:",public"`
}

func (c *TinyMultiply) Define(api frontend.API) error {
	xx := api.Mul(c.X, c.X)
	api.AssertIsEqual(xx, c.Y)
	return nil
}

func main() {
	outputDir := flag.String("output_dir", "../data/tiny_multiply", "Output directory")
	flag.Parse()

	if err := os.MkdirAll(*outputDir, 0o755); err != nil {
		panic(err)
	}

	// --- Compile circuit ---
	var circuit TinyMultiply
	ccs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &circuit)
	if err != nil {
		panic(fmt.Errorf("compile: %w", err))
	}

	r1csTyped := ccs.(*cs_bn254.R1CS)

	// --- Setup ---
	pk, vk, err := groth16.Setup(ccs)
	if err != nil {
		panic(fmt.Errorf("setup: %w", err))
	}
	pkTyped := pk.(*groth16bn254.ProvingKey)
	vkTyped := vk.(*groth16bn254.VerifyingKey)

	// --- Create witness: x=3, y=9 ---
	assignment := &TinyMultiply{X: 3, Y: 9}
	witness, err := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
	if err != nil {
		panic(fmt.Errorf("witness: %w", err))
	}

	// --- Solve to get full solution ---
	sol, err := r1csTyped.Solve(witness)
	if err != nil {
		panic(fmt.Errorf("solve: %w", err))
	}
	solution := sol.(*cs_bn254.R1CSSolution)

	// R1CSSolution.W is the full witness vector (all wires)
	// R1CSSolution.A, B, C are the constraint evaluation vectors
	nbConstraints := ccs.GetNbConstraints()
	nbWires := len(solution.W)
	nbPublic := ccs.GetNbPublicVariables()
	nbSecret := ccs.GetNbSecretVariables()
	nbInternal := ccs.GetNbInternalVariables()

	// Domain size: next power of 2 >= nbConstraints
	domainSize := 1
	for domainSize < nbConstraints {
		domainSize <<= 1
	}
	if domainSize < 2 {
		domainSize = 2
	}

	fmt.Printf("Circuit: %d wires, %d constraints, domain_size=%d\n",
		nbWires, nbConstraints, domainSize)
	fmt.Printf("  public=%d, secret=%d, internal=%d\n",
		nbPublic, nbSecret, nbInternal)

	// --- Export ---
	d := *outputDir

	// metadata.json
	meta := map[string]int{
		"num_wires":       nbWires,
		"num_public":      nbPublic,
		"num_secret":      nbSecret,
		"num_internal":    nbInternal,
		"num_constraints": nbConstraints,
		"domain_size":     domainSize,
	}
	writeJSON(filepath.Join(d, "metadata.json"), meta)

	// witness_full.bin — full witness (all wires, Montgomery form)
	writeFieldElements(filepath.Join(d, "witness_full.bin"), solution.W)

	// solution_a/b/c.bin — constraint evaluations
	writeFieldElements(filepath.Join(d, "solution_a.bin"), solution.A)
	writeFieldElements(filepath.Join(d, "solution_b.bin"), solution.B)
	writeFieldElements(filepath.Join(d, "solution_c.bin"), solution.C)

	// PK points — uncompact A and B arrays
	infinityA := pkTyped.InfinityA
	infinityB := pkTyped.InfinityB

	aG1 := uncompactG1(pkTyped.G1.A, infinityA, nbWires)
	bG1 := uncompactG1(pkTyped.G1.B, infinityB, nbWires)
	bG2 := uncompactG2(pkTyped.G2.B, infinityB, nbWires)

	writeG1Points(filepath.Join(d, "pk_a_g1.bin"), aG1)
	writeG1Points(filepath.Join(d, "pk_b_g1.bin"), bG1)
	writeG2Points(filepath.Join(d, "pk_b_g2.bin"), bG2)
	writeG1Points(filepath.Join(d, "pk_k_g1.bin"), pkTyped.G1.K)
	writeG1Points(filepath.Join(d, "pk_z_g1.bin"), pkTyped.G1.Z)
	writeG1Point(filepath.Join(d, "pk_delta_g1.bin"), pkTyped.G1.Delta)
	writeG2Point(filepath.Join(d, "pk_delta_g2.bin"), pkTyped.G2.Delta)

	// Infinity masks
	writeBoolMask(filepath.Join(d, "infinity_a.bin"), infinityA)
	writeBoolMask(filepath.Join(d, "infinity_b.bin"), infinityB)

	// VK points
	writeG1Point(filepath.Join(d, "vk_alpha_g1.bin"), vkTyped.G1.Alpha)
	writeG1Point(filepath.Join(d, "vk_beta_g1.bin"), pkTyped.G1.Beta)
	writeG2Point(filepath.Join(d, "vk_beta_g2.bin"), vkTyped.G2.Beta)
	writeG2Point(filepath.Join(d, "vk_gamma_g2.bin"), vkTyped.G2.Gamma)
	writeG1Points(filepath.Join(d, "vk_ic.bin"), vkTyped.G1.K)

	// R1CS in COO format
	r1cs := r1csTyped.GetR1Cs()
	writeCOO(filepath.Join(d, "r1cs_a.bin"), r1cs, 'L', r1csTyped.Coefficients)
	writeCOO(filepath.Join(d, "r1cs_b.bin"), r1cs, 'R', r1csTyped.Coefficients)
	writeCOO(filepath.Join(d, "r1cs_c.bin"), r1cs, 'O', r1csTyped.Coefficients)

	// Levels
	levels := r1csTyped.Levels
	if len(levels) == 0 {
		levels = [][]uint32{{0}}
		for i := 1; i < nbConstraints; i++ {
			levels[0] = append(levels[0], uint32(i))
		}
	}

	levelSizes := make([]uint32, len(levels))
	for i, l := range levels {
		levelSizes[i] = uint32(len(l))
	}
	writeUint32s(filepath.Join(d, "r1cs_level_sizes.bin"), levelSizes)

	var levelOrder []uint32
	for _, l := range levels {
		levelOrder = append(levelOrder, l...)
	}
	writeUint32s(filepath.Join(d, "r1cs_level_order.bin"), levelOrder)

	// Level unknowns: (side uint8, wireID uint32) per constraint
	writeUnknowns(filepath.Join(d, "r1cs_level_unknowns.bin"), nbConstraints)

	// --- Generate proof using gnark native prover ---
	proof, err := groth16.Prove(ccs, pk, witness)
	if err != nil {
		panic(fmt.Errorf("prove: %w", err))
	}
	proofTyped := proof.(*groth16bn254.Proof)

	// Verify proof natively to confirm correctness
	publicWitness, err := witness.Public()
	if err != nil {
		panic(fmt.Errorf("public witness: %w", err))
	}
	if err := groth16.Verify(proof, vk, publicWitness); err != nil {
		panic(fmt.Errorf("native verify failed: %w", err))
	}
	fmt.Println("Native gnark verify: PASS")

	// Export proof as snarkjs-compatible JSON
	proofJSON := map[string]interface{}{
		"pi_a":     g1ToJSON(proofTyped.Ar),
		"pi_b":     g2ToJSON(proofTyped.Bs),
		"pi_c":     g1ToJSON(proofTyped.Krs),
		"protocol": "groth16",
		"curve":    "bn128",
	}
	writeJSON(filepath.Join(d, "proof.json"), proofJSON)

	// Export public signals (gnark convention: includes constant ONE wire).
	// gnark wire ordering: [ONE, public..., secret..., internal...]
	// The Python verifier auto-detects gnark style when
	// len(public_signals) == len(IC).
	publicSignals := make([]string, nbPublic)
	for i := 0; i < nbPublic; i++ {
		var v big.Int
		solution.W[i].BigInt(&v)
		publicSignals[i] = v.String()
	}
	writeJSON(filepath.Join(d, "public.json"), publicSignals)

	fmt.Printf("Exported fixture to %s\n", d)
}

func g1ToJSON(pt bn254.G1Affine) []string {
	var x, y big.Int
	pt.X.BigInt(&x)
	pt.Y.BigInt(&y)
	return []string{x.String(), y.String(), "1"}
}

func g2ToJSON(pt bn254.G2Affine) [3]interface{} {
	var a0, a1 big.Int
	pt.X.A0.BigInt(&a0)
	pt.X.A1.BigInt(&a1)
	xPart := []string{a0.String(), a1.String()}
	pt.Y.A0.BigInt(&a0)
	pt.Y.A1.BigInt(&a1)
	yPart := []string{a0.String(), a1.String()}
	return [3]interface{}{xPart, yPart, []string{"1", "0"}}
}

func uncompactG1(compacted []bn254.G1Affine, infinity []bool, n int) []bn254.G1Affine {
	result := make([]bn254.G1Affine, n)
	idx := 0
	for i := 0; i < n; i++ {
		if !infinity[i] {
			result[i] = compacted[idx]
			idx++
		}
	}
	return result
}

func uncompactG2(compacted []bn254.G2Affine, infinity []bool, n int) []bn254.G2Affine {
	result := make([]bn254.G2Affine, n)
	idx := 0
	for i := 0; i < n; i++ {
		if !infinity[i] {
			result[i] = compacted[idx]
			idx++
		}
	}
	return result
}

func writeJSON(path string, v interface{}) {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		panic(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		panic(err)
	}
}

func writeFieldElements(path string, elems []fr.Element) {
	f, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	// Write raw Montgomery form: fr.Element is [4]uint64 in Montgomery
	// representation.  Write the limbs directly in little-endian byte order
	// to match the zk_dtypes bn254_sf_mont memory layout.
	for _, e := range elems {
		if err := binary.Write(f, binary.LittleEndian, e); err != nil {
			panic(err)
		}
	}
}

func writeG1Points(path string, pts []bn254.G1Affine) {
	f, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	for _, pt := range pts {
		writeG1ToFile(f, pt)
	}
}

func writeG1Point(path string, pt bn254.G1Affine) {
	f, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	writeG1ToFile(f, pt)
}

func writeG1ToFile(f *os.File, pt bn254.G1Affine) {
	var xBig, yBig big.Int
	pt.X.BigInt(&xBig)
	pt.Y.BigInt(&yBig)
	writeLE32(f, &xBig)
	writeLE32(f, &yBig)
}

func writeG2Points(path string, pts []bn254.G2Affine) {
	f, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	for _, pt := range pts {
		writeG2ToFile(f, pt)
	}
}

func writeG2Point(path string, pt bn254.G2Affine) {
	f, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	writeG2ToFile(f, pt)
}

func writeG2ToFile(f *os.File, pt bn254.G2Affine) {
	var a0, a1 big.Int
	pt.X.A0.BigInt(&a0)
	pt.X.A1.BigInt(&a1)
	writeLE32(f, &a0)
	writeLE32(f, &a1)
	pt.Y.A0.BigInt(&a0)
	pt.Y.A1.BigInt(&a1)
	writeLE32(f, &a0)
	writeLE32(f, &a1)
}

func writeLE32(f *os.File, v *big.Int) {
	var buf [32]byte
	b := v.Bytes() // big-endian
	for i := 0; i < len(b); i++ {
		buf[i] = b[len(b)-1-i]
	}
	if _, err := f.Write(buf[:]); err != nil {
		panic(err)
	}
}

func writeBoolMask(path string, mask []bool) {
	data := make([]byte, len(mask))
	for i, v := range mask {
		if v {
			data[i] = 1
		}
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		panic(err)
	}
}

// writeCOO writes R1CS matrix in COO format: per entry 4B row + 4B col + 32B val.
func writeCOO(path string, constraints []constraint.R1C, side byte, coeffs []fr.Element) {
	f, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	for row, c := range constraints {
		var terms constraint.LinearExpression
		switch side {
		case 'L':
			terms = c.L
		case 'R':
			terms = c.R
		case 'O':
			terms = c.O
		}
		for _, term := range terms {
			if err := binary.Write(f, binary.LittleEndian, uint32(row)); err != nil {
				panic(err)
			}
			if err := binary.Write(f, binary.LittleEndian, uint32(term.WireID())); err != nil {
				panic(err)
			}
			// Write coefficient in raw Montgomery form
			coeff := coeffs[term.CoeffID()]
			if err := binary.Write(f, binary.LittleEndian, coeff); err != nil {
				panic(err)
			}
		}
	}
}

func writeUint32s(path string, vals []uint32) {
	f, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	if err := binary.Write(f, binary.LittleEndian, vals); err != nil {
		panic(err)
	}
}

func writeUnknowns(path string, n int) {
	f, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	for i := 0; i < n; i++ {
		f.Write([]byte{255})
		binary.Write(f, binary.LittleEndian, uint32(0))
	}
}
