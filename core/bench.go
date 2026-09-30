// Package core implements the layered clipping / Gaussian-noise aggregation /
// layer-wise scaling numerical experiment. All computation is deterministic
// in (Config, Seed); no global RNG state is used.
package core

import (
	"errors"
	"math"
	"strconv"
)

// MaxRepeats caps the repeats accepted by Run.
const MaxRepeats = 10000

// BudgetMode selects how the total clipping budget is split across layers.
type BudgetMode string

const (
	// BudgetEqual gives every layer b = B/sqrt(L), so sum_l b_l^2 = B^2.
	BudgetEqual BudgetMode = "equal"
	// BudgetProportional makes b_l^2 proportional to layer dimension d_l:
	// b_l = B*sqrt(d_l/D), D = sum d_l, so sum_l b_l^2 = B^2.
	BudgetProportional BudgetMode = "proportional"
)

// Layer is one layer: parameter vector and one update vector per client.
type Layer struct {
	Name    string      `json:"name"`
	Params  []float64   `json:"params"`
	Clients [][]float64 `json:"clients"`
}

// Config is the full editable experiment configuration.
type Config struct {
	Layers []Layer `json:"layers"`

	Budget     float64    `json:"budget"`     // total clipping budget B (> 0)
	BudgetMode BudgetMode `json:"budgetMode"` // equal | proportional

	Sigma   float64 `json:"sigma"`   // std-dev of per-coordinate Gaussian noise (>= 0)
	Seed    int64   `json:"seed"`    // base RNG seed
	Repeats int     `json:"repeats"` // number of independent repetitions (>= 1)

	ScaleCap float64 `json:"scaleCap"` // upper bound on the scale coefficient (> 0)
	// FallbackScale is used when the aggregate norm is zero (see ScaleCoeff).
	FallbackScale float64 `json:"fallbackScale"`
}

// ClientLayerStat summarizes clipping of one client on one layer.
type ClientLayerStat struct {
	NormBefore float64 `json:"normBefore"`
	NormAfter  float64 `json:"normAfter"`
	// Clipped is true when the update norm exceeded the layer budget.
	Clipped bool `json:"clipped"`
	// Ratio is the fraction of the original norm removed by clipping
	// (0 when no clipping happened or the input norm was zero).
	Ratio float64 `json:"ratio"`
}

// RepLayer is one repetition's result for one layer.
type RepLayer struct {
	AggNorm float64   `json:"aggNorm"` // noisy (unscaled) aggregate norm
	Scale   float64   `json:"scale"`   // applied scale coefficient
	Update  []float64 `json:"update"`  // final, scaled update
	Raw     []float64 `json:"raw"`     // noisy aggregate before scaling
}

// LayerResult is the complete per-layer result block.
type LayerResult struct {
	Index  int     `json:"index"`
	Name   string  `json:"name"`
	Dim    int     `json:"dim"`
	Budget float64 `json:"budget"`

	ParamNorm float64 `json:"paramNorm"`

	// Per-client clipping summary (same indices as Config clients).
	Clients []ClientLayerStat `json:"clients"`

	// Deterministic, noise-free reference.
	RefAggNorm float64   `json:"refAggNorm"`
	RefScale   float64   `json:"refScale"`
	RefUpdate  []float64 `json:"refUpdate"`

	// Per-repetition results.
	Reps []RepLayer `json:"reps"`

	// Statistics over repetitions of the final scaled update.
	MeanUpdate []float64 `json:"meanUpdate"`
	VarUpdate  []float64 `json:"varUpdate"`
	MaxAbsErr  float64   `json:"maxAbsErr"`
	RMSE       float64   `json:"rmse"`

	// Statistics over repetitions of the unscaled noisy aggregate.
	MeanRaw []float64 `json:"meanRaw"`
	VarRaw  []float64 `json:"varRaw"`

	// Statistics of the scale coefficient across repetitions.
	MeanScale float64 `json:"meanScale"`
	VarScale  float64 `json:"varScale"`
}

// Report is the output of one experiment run.
type Report struct {
	Layers []LayerResult `json:"layers"`
	N      int           `json:"numClients"`
	Config Config        `json:"config"`
}

// ErrEmptyConfig is returned by Validate when there is nothing to compute.
var ErrEmptyConfig = errors.New("configuration must contain at least one layer")

// Validate checks the whole configuration. Invalid configs are rejected
// wholesale (no partial report is produced) so the UI can keep showing the
// last valid result.
func (c *Config) Validate() error {
	if len(c.Layers) == 0 {
		return ErrEmptyConfig
	}
	if !(c.Budget > 0) || math.IsNaN(c.Budget) || math.IsInf(c.Budget, 0) {
		return errors.New("total budget must be a finite positive number")
	}
	switch c.BudgetMode {
	case BudgetEqual, BudgetProportional:
	default:
		return errors.New("budget mode must be \"equal\" or \"proportional\"")
	}
	if math.IsNaN(c.Sigma) || math.IsInf(c.Sigma, 0) || c.Sigma < 0 {
		return errors.New("noise sigma must be a finite non-negative number")
	}
	if c.Repeats < 1 {
		return errors.New("repeats must be at least 1")
	}
	if c.Repeats > MaxRepeats {
		return errors.New("repeats exceed the maximum of 10000")
	}
	if !(c.ScaleCap > 0) || math.IsNaN(c.ScaleCap) || math.IsInf(c.ScaleCap, 0) {
		return errors.New("scale cap must be a finite positive number")
	}
	if !(c.FallbackScale >= 0) || math.IsNaN(c.FallbackScale) || math.IsInf(c.FallbackScale, 0) {
		return errors.New("fallback scale must be a finite non-negative number")
	}
	if c.FallbackScale > c.ScaleCap {
		return errors.New("fallback scale must not exceed the scale cap")
	}

	n := len(c.Layers[0].Clients)
	if n == 0 {
		return errors.New("there must be at least one client update per layer")
	}
	for li, l := range c.Layers {
		if len(l.Params) == 0 {
			return errors.New("layer " + layerName(l, li) + " has an empty parameter vector")
		}
		if err := allFinite("layer "+layerName(l, li)+" params", l.Params); err != nil {
			return err
		}
		if len(l.Clients) != n {
			return errors.New("layer " + layerName(l, li) + " has " +
				"a client count inconsistent with the first layer")
		}
		for _, u := range l.Clients {
			if len(u) != len(l.Params) {
				return errors.New("layer " + layerName(l, li) +
					": client update shape does not match the parameter shape")
			}
			if err := allFinite("layer "+layerName(l, li)+" client update", u); err != nil {
				return err
			}
		}
	}
	return nil
}

func layerName(l Layer, i int) string {
	if l.Name != "" {
		return l.Name
	}
	return "L" + strconv.Itoa(i)
}

func allFinite(what string, v []float64) error {
	for _, x := range v {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return errors.New(what + " contains a non-finite value")
		}
	}
	return nil
}

// AllocateBudgets splits the total budget across L layers.
// Both modes satisfy sum budgets[l]^2 == budget^2.
func AllocateBudgets(budget float64, dims []int, mode BudgetMode) []float64 {
	L := len(dims)
	out := make([]float64, L)
	switch mode {
	case BudgetProportional:
		totalDim := 0
		for _, d := range dims {
			totalDim += d
		}
		for i, d := range dims {
			out[i] = budget * math.Sqrt(float64(d)/float64(totalDim))
		}
	default: // BudgetEqual
		b := budget / math.Sqrt(float64(L))
		for i := range out {
			out[i] = b
		}
	}
	return out
}

func norm(v []float64) float64 {
	var s float64
	for _, x := range v {
		s += x * x
	}
	return math.Sqrt(s)
}

// ClipByNorm clips v in place to Euclidean norm at most b.
// It returns the norm before clipping, the norm after clipping, whether
// clipping occurred and the fraction of the original norm removed.
func ClipByNorm(v []float64, b float64) (before, after float64, clipped bool, ratio float64) {
	n := norm(v)
	before = n
	if n > b {
		f := b / n
		for i := range v {
			v[i] *= f
		}
		after = b
		return before, after, true, (n - b) / n
	}
	return before, n, false, 0
}

// ScaleCoeff returns min(paramNorm/aggNorm, cap). When aggNorm is zero the
// ratio is undefined; the explicit finite fallback is used instead (it must be
// non-negative and not exceed cap), so the result is always finite and no
// division by zero occurs.
func ScaleCoeff(paramNorm, aggNorm, capV, fallback float64) float64 {
	if aggNorm == 0 {
		return fallback
	}
	s := paramNorm / aggNorm
	if s > capV {
		return capV
	}
	return s
}
