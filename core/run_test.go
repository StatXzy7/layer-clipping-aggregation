package core

import (
	"math"
	"testing"
)

func baseCfg() Config {
	return Config{
		Layers: []Layer{{
			Name:    "L0",
			Params:  []float64{6, 8},
			Clients: [][]float64{{3, 4}, {0, 0}},
		}},
		Budget:        5,
		BudgetMode:    BudgetEqual,
		Sigma:         0,
		Seed:          42,
		Repeats:       1,
		ScaleCap:      10,
		FallbackScale: 1,
	}
}

// Hand-computed end-to-end check with noise. The Gaussian draws were
// independently reproduced in Python with the documented splitmix64 +
// Box-Muller generator:
//
//	seed mix(42,0) draws (per layer, client-major, 2 coords/client):
//	  0.5430526513692419, 0.8600721962080228,
//	  1.8082992021623316, 0.5628515712662702
//
// Clipping: c0 (3,4) norm 5 == budget, untouched; c1 zero.
// refAgg = (1.5, 2), norm 2.5; param norm 10; reference scale 4 and
// reference update exactly (6, 8).
// Noisy aggregate (sigma 0.5, n 2):
//
//	x = 1.5 + 0.5*(g00+g10)/2 = 2.0878379633828934
//	y = 2.0 + 0.5*(g01+g11)/2 = 2.3557309418685732
//	norm = 3.1477826532052404; scale = 10/norm = 3.1768394141881005
func TestRunWithHandComputedNoise(t *testing.T) {
	cfg := baseCfg()
	cfg.Sigma = 0.5
	cfg.Repeats = 1
	rep, err := Run(cfg)
	if err != nil {
		t.Fatal(err)
	}
	l := rep.Layers[0]

	approxEq(t, l.Clients[0].NormBefore, 5, 0, "c0 before")
	if l.Clients[0].Clipped {
		t.Fatal("c0 must not be clipped")
	}
	approxEq(t, l.RefAggNorm, 2.5, 0, "refAggNorm")
	approxEq(t, l.RefScale, 4, 0, "refScale")
	approxEq(t, l.RefUpdate[0], 6, 0, "ref x")
	approxEq(t, l.RefUpdate[1], 8, 0, "ref y")

	r := l.Reps[0]
	approxEq(t, r.Raw[0], 2.0878379633828934, 1e-14, "raw x")
	approxEq(t, r.Raw[1], 2.3557309418685732, 1e-14, "raw y")
	approxEq(t, r.AggNorm, 3.1477826532052404, 1e-14, "aggNorm")
	approxEq(t, r.Scale, 3.1768394141881005, 1e-14, "scale")
	approxEq(t, r.Update[0], 6.6327259325129875, 1e-13, "final x")
	approxEq(t, r.Update[1], 7.4837789053505404, 1e-13, "final y")
}

func TestZeroNoiseIsExactlyDeterministic(t *testing.T) {
	cfg := baseCfg()
	cfg.Repeats = 7
	r1, err := Run(cfg)
	if err != nil {
		t.Fatal(err)
	}
	// A different seed must not matter when sigma == 0.
	cfg.Seed = 999983
	r2, err := Run(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for i, l := range r1.Layers {
		for k, rp := range l.Reps {
			o := r2.Layers[i].Reps[k]
			if rp.Raw[0] != o.Raw[0] || rp.Raw[1] != o.Raw[1] ||
				rp.Update[0] != o.Update[0] || rp.Update[1] != o.Update[1] ||
				rp.Scale != o.Scale {
				t.Fatalf("zero-noise result not exact at layer %d rep %d", i, k)
			}
			// Every repetition equals the noise-free reference exactly.
			if rp.Update[0] != l.RefUpdate[0] || rp.Update[1] != l.RefUpdate[1] ||
				rp.Raw[0] != 1.5 || rp.Raw[1] != 2 || rp.Scale != 4 {
				t.Fatalf("rep %d deviates from reference: %+v", k, rp)
			}
		}
		if l.RMSE != 0 || l.MaxAbsErr != 0 {
			t.Fatalf("zero-noise error metrics nonzero: %v %v", l.RMSE, l.MaxAbsErr)
		}
		for _, v := range l.VarUpdate {
			if v != 0 {
				t.Fatalf("zero-noise variance nonzero: %v", l.VarUpdate)
			}
		}
	}
}

func TestFixedSeedReproducible(t *testing.T) {
	cfg := baseCfg()
	cfg.Sigma = 1.0
	cfg.Repeats = 25
	a, err := Run(cfg)
	if err != nil {
		t.Fatal(err)
	}
	b, err := Run(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for i := range a.Layers {
		for k := range a.Layers[i].Reps {
			ra, rb := a.Layers[i].Reps[k], b.Layers[i].Reps[k]
			if ra.Raw[0] != rb.Raw[0] || ra.Raw[1] != rb.Raw[1] ||
				ra.Update[0] != rb.Update[0] || ra.Update[1] != rb.Update[1] {
				t.Fatalf("non-reproducible run layer %d rep %d", i, k)
			}
		}
	}

	// Different seed must (overwhelmingly likely) change the draws: assert at
	// least one repetition differs, proving statistics are tied to the seed.
	cfg.Seed = cfg.Seed + 1
	c, err := Run(cfg)
	if err != nil {
		t.Fatal(err)
	}
	diff := false
	for k := range a.Layers[0].Reps {
		if a.Layers[0].Reps[k].Raw[0] != c.Layers[0].Reps[k].Raw[0] {
			diff = true
		}
	}
	if !diff {
		t.Fatal("seed change did not affect any repetition")
	}
}

// Clipping in a full Run, proportional budgets: dims [1,3], B=2 gives the
// dim-3 layer budget 2*sqrt(3/4) = sqrt(3). c0 (1,2,2) norm 3 clips to
// sqrt(3) with ratio 1-sqrt(3)/3; the zero client stays zero.
func TestRunClippingAndAggregation(t *testing.T) {
	cfg := Config{
		Layers: []Layer{
			{Name: "a", Params: []float64{2}, Clients: [][]float64{{3}, {1}}},
			{Name: "b", Params: []float64{0, 0, 0},
				Clients: [][]float64{{1, 2, 2}, {0, 0, 0}}},
		},
		Budget:        2,
		BudgetMode:    BudgetProportional,
		Repeats:       1,
		ScaleCap:      10,
		FallbackScale: 1,
	}
	rep, err := Run(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if rep.N != 2 {
		t.Fatalf("num clients = %d", rep.N)
	}
	l0, l1 := rep.Layers[0], rep.Layers[1]
	approxEq(t, l0.Budget, 1, 1e-15, "budget dim1") // 2*sqrt(1/4)
	approxEq(t, l1.Budget, math.Sqrt(3), 1e-15, "budget dim3")

	c0 := l1.Clients[0]
	approxEq(t, c0.NormBefore, 3, 0, "before")
	approxEq(t, c0.NormAfter, math.Sqrt(3), 1e-15, "after")
	if !c0.Clipped {
		t.Fatal("expected clipped")
	}
	approxEq(t, c0.Ratio, 1-math.Sqrt(3)/3, 1e-15, "ratio")

	// refAgg = clipped(c0)/2; param norm 0 -> scale 0 -> final zeros,
	// even though the aggregate is nonzero.
	approxEq(t, l1.RefAggNorm, math.Sqrt(3)/2, 1e-15, "refAggNorm")
	approxEq(t, l1.RefScale, 0, 0, "refScale zero")
	approxEq(t, l1.RefUpdate[0], 0, 0, "refUpdate0")
	approxEq(t, l1.RefUpdate[1], 0, 0, "refUpdate1")

	// Layer 0: {3} clips to budget 1, {1} already equals the budget; aggregate
	// (1+1)/2 = 1,
	// param norm 2 -> scale 2 -> final 2.
	if !l0.Clients[0].Clipped {
		t.Fatal("layer-0 client {3} should clip")
	}
	if l0.Clients[1].Clipped {
		t.Fatal("layer-0 client {1} must not clip (norm == budget)")
	}
	approxEq(t, l0.RefAggNorm, 1, 0, "l0 agg norm")
	approxEq(t, l0.RefScale, 2, 0, "l0 scale")
	approxEq(t, l0.RefUpdate[0], 2, 0, "l0 final")
}

// All clients zero on a layer with nonzero parameters: aggregate norm zero,
// fallback scale must be used and the update finite.
func TestRunZeroAggregateUsesFallback(t *testing.T) {
	cfg := Config{
		Layers: []Layer{{
			Name:    "z",
			Params:  []float64{3, 4},
			Clients: [][]float64{{0, 0}, {0, 0}},
		}},
		Budget: 2, BudgetMode: BudgetEqual, Repeats: 4,
		ScaleCap: 10, FallbackScale: 0.5,
	}
	rep, err := Run(cfg)
	if err != nil {
		t.Fatal(err)
	}
	l := rep.Layers[0]
	approxEq(t, l.RefAggNorm, 0, 0, "zero agg norm")
	approxEq(t, l.RefScale, 0.5, 0, "fallback used")
	for k, rp := range l.Reps {
		if rp.AggNorm != 0 || rp.Scale != 0.5 {
			t.Fatalf("rep %d not zero/fallback: %+v", k, rp)
		}
		for _, x := range rp.Update {
			if x != 0 || math.IsNaN(x) || math.IsInf(x, 0) {
				t.Fatalf("non-finite/nonzero final update: %v", rp.Update)
			}
		}
	}
}

// With many zero-mean repetitions the sample mean tracks the deterministic
// reference and the empirical raw variance approaches sigma^2/n^2.
func TestRunNoiseStatistics(t *testing.T) {
	cfg := baseCfg()
	cfg.Sigma = 0.5
	cfg.Repeats = 4000
	rep, err := Run(cfg)
	if err != nil {
		t.Fatal(err)
	}
	l := rep.Layers[0]
	// Unscaled aggregate variance per coordinate: sigma^2/n = 0.25/2.
	// (n independent noises averaged -> sigma^2/n).
	for k, v := range l.VarRaw {
		want := 0.25 / 2.0
		if math.Abs(v-want) > 0.02 {
			t.Fatalf("VarRaw[%d]=%v want ~%v", k, v, want)
		}
	}
	// Mean raw close to refAgg (1.5, 2).
	if math.Abs(l.MeanRaw[0]-1.5) > 0.05 || math.Abs(l.MeanRaw[1]-2) > 0.05 {
		t.Fatalf("MeanRaw off: %v", l.MeanRaw)
	}
	if l.RMSE <= 0 {
		t.Fatal("expected positive RMSE with noise")
	}
}
