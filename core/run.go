package core

import "math"

// Run executes the full experiment. The configuration is validated first;
// every report is computed from scratch from the given config and seed, so
// changed parameters never reuse statistics drawn for an older seed.
func Run(cfg Config) (*Report, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	L := len(cfg.Layers)
	n := len(cfg.Layers[0].Clients)
	dims := make([]int, L)
	for i, l := range cfg.Layers {
		dims[i] = len(l.Params)
	}
	budgets := AllocateBudgets(cfg.Budget, dims, cfg.BudgetMode)

	rep := &Report{Layers: make([]LayerResult, L), N: n, Config: cfg}

	for li, layer := range cfg.Layers {
		d := dims[li]
		lr := LayerResult{
			Index:     li,
			Name:      layerName(layer, li),
			Dim:       d,
			Budget:    budgets[li],
			ParamNorm: norm(layer.Params),
			Clients:   make([]ClientLayerStat, n),
			Reps:      make([]RepLayer, cfg.Repeats),
		}

		// Per-client Euclidean clipping, then equal-weight mean of the
		// clipped updates gives the noise-free reference aggregate.
		refAgg := make([]float64, d)
		for ci := 0; ci < n; ci++ {
			u := append([]float64(nil), layer.Clients[ci]...)
			before, after, clipped, ratio := ClipByNorm(u, budgets[li])
			lr.Clients[ci] = ClientLayerStat{
				NormBefore: before,
				NormAfter:  after,
				Clipped:    clipped,
				Ratio:      ratio,
			}
			for k := 0; k < d; k++ {
				refAgg[k] += u[k] / float64(n)
			}
		}
		lr.RefAggNorm = norm(refAgg)
		lr.RefScale = ScaleCoeff(lr.ParamNorm, lr.RefAggNorm,
			cfg.ScaleCap, cfg.FallbackScale)
		lr.RefUpdate = scale(refAgg, lr.RefScale)

		// Independent repetitions. Noise draw order within a repetition is
		// layer -> client -> coordinate, so each coordinate gets one
		// independent N(0, sigma^2) draw added after clipping.
		for r := 0; r < cfg.Repeats; r++ {
			agg := make([]float64, d)
			if cfg.Sigma > 0 {
				rng := newGaussRNG(mixSeed(cfg.Seed, r))
				// Noise draws are interleaved across layers so the draw
				// order is rep -> layer -> client -> coordinate.
				for li2 := 0; li2 < li; li2++ {
					for ci := 0; ci < n; ci++ {
						for k := 0; k < dims[li2]; k++ {
							rng.gauss()
						}
					}
				}
				for ci := 0; ci < n; ci++ {
					for k := 0; k < d; k++ {
						agg[k] += refAgg[k] + cfg.Sigma*rng.gauss()
					}
				}
				for k := 0; k < d; k++ {
					agg[k] /= float64(n)
				}
			} else {
				// Zero noise: skip the RNG entirely and copy the
				// deterministic aggregate bit-for-bit.
				copy(agg, refAgg)
			}
			aNorm := norm(agg)
			s := ScaleCoeff(lr.ParamNorm, aNorm, cfg.ScaleCap, cfg.FallbackScale)
			lr.Reps[r] = RepLayer{
				AggNorm: aNorm,
				Scale:   s,
				Raw:     append([]float64(nil), agg...),
				Update:  scale(agg, s),
			}
		}

		finalizeStats(&lr, cfg.Repeats)
		rep.Layers[li] = lr
	}
	return rep, nil
}

func scale(v []float64, s float64) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = x * s
	}
	return out
}

// finalizeStats computes mean/population-variance over repetitions of the final
// update, the unscaled aggregate and the scale coefficient, plus error metrics
// against the noise-free reference.
func finalizeStats(lr *LayerResult, reps int) {
	d := lr.Dim
	mean := make([]float64, d)
	meanRaw := make([]float64, d)
	var meanScale float64
	for _, rp := range lr.Reps {
		for k := 0; k < d; k++ {
			mean[k] += rp.Update[k]
			meanRaw[k] += rp.Raw[k]
		}
		meanScale += rp.Scale
	}
	for k := 0; k < d; k++ {
		mean[k] /= float64(reps)
		meanRaw[k] /= float64(reps)
	}
	meanScale /= float64(reps)

	variance := make([]float64, d)
	varRaw := make([]float64, d)
	var varScale float64
	var sse float64
	maxAbs := 0.0
	for _, rp := range lr.Reps {
		for k := 0; k < d; k++ {
			dv := rp.Update[k] - mean[k]
			variance[k] += dv * dv
			dr := rp.Raw[k] - meanRaw[k]
			varRaw[k] += dr * dr
			e := rp.Update[k] - lr.RefUpdate[k]
			sse += e * e
			if a := math.Abs(e); a > maxAbs {
				maxAbs = a
			}
		}
		ds := rp.Scale - meanScale
		varScale += ds * ds
	}
	m := float64(reps)
	for k := 0; k < d; k++ {
		variance[k] /= m
		varRaw[k] /= m
	}
	varScale /= m

	lr.MeanUpdate = mean
	lr.VarUpdate = variance
	lr.MeanRaw = meanRaw
	lr.VarRaw = varRaw
	lr.MeanScale = meanScale
	lr.VarScale = varScale
	lr.MaxAbsErr = maxAbs
	lr.RMSE = math.Sqrt(sse / (m * float64(d)))
}
