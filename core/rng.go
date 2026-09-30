package core

import "math"

// gaussRNG is a small self-contained deterministic pseudo-random generator.
// It uses splitmix64 to produce uniform bits and the Box-Muller transform for
// standard-normal samples. Having an explicit implementation (rather than
// math/rand) makes the generated numbers reproducible across Go toolchains and
// lets tests assert exact hand-computed values.
type gaussRNG struct {
	state uint64
	spare float64
	has   bool
}

func newGaussRNG(seed uint64) *gaussRNG {
	return &gaussRNG{state: seed}
}

// nextUint64 is the splitmix64 step.
func (r *gaussRNG) nextUint64() uint64 {
	r.state += 0x9e3779b97f4a7c15
	z := r.state
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

// float64 returns a uniform value in [0, 1).
func (r *gaussRNG) float64() float64 {
	return float64(r.nextUint64()>>11) * 0x1.0p-53
}

// gauss returns a standard-normal sample (Box-Muller).
func (r *gaussRNG) gauss() float64 {
	if r.has {
		r.has = false
		return r.spare
	}
	u1 := r.float64()
	for u1 == 0 {
		u1 = r.float64()
	}
	u2 := r.float64()
	mag := math.Sqrt(-2 * math.Log(u1))
	a := 2 * math.Pi * u2
	r.spare = mag * math.Sin(a)
	r.has = true
	return mag * math.Cos(a)
}

// mixSeed combines the base seed with a repetition index without sharing RNG
// state between repetitions.
func mixSeed(base int64, rep int) uint64 {
	x := uint64(base) ^ (uint64(rep) * 0x9e3779b97f4a7c15)
	x ^= x >> 30
	x *= 0xbf58476d1ce4e5b9
	x ^= x >> 27
	x *= 0x94d049bb133111eb
	x ^= x >> 31
	return x
}
