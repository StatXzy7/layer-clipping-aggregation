package core

import (
	"errors"
	"math"
	"testing"
)

func validConfig() Config {
	return Config{
		Layers: []Layer{{
			Name:    "ok",
			Params:  []float64{1, 2},
			Clients: [][]float64{{1, 0}, {0, 1}},
		}},
		Budget:        4,
		BudgetMode:    BudgetEqual,
		Sigma:         0.1,
		Seed:          1,
		Repeats:       2,
		ScaleCap:      10,
		FallbackScale: 1,
	}
}

func TestValidateAcceptsBase(t *testing.T) {
	c := validConfig()
	if err := c.Validate(); err != nil {
		t.Fatalf("base config rejected: %v", err)
	}
	if _, err := Run(validConfig()); err != nil {
		t.Fatalf("base run rejected: %v", err)
	}
}

func TestValidateRejectsInvalid(t *testing.T) {
	cases := []struct {
		name string
		mut  func(*Config)
	}{
		{"no layers", func(c *Config) { c.Layers = nil }},
		{"empty client list", func(c *Config) { c.Layers[0].Clients = nil }},
		{"zero budget", func(c *Config) { c.Budget = 0 }},
		{"negative budget", func(c *Config) { c.Budget = -1 }},
		{"NaN budget", func(c *Config) { c.Budget = math.NaN() }},
		{"Inf budget", func(c *Config) { c.Budget = math.Inf(1) }},
		{"bad mode", func(c *Config) { c.BudgetMode = "weird" }},
		{"negative sigma", func(c *Config) { c.Sigma = -0.1 }},
		{"NaN sigma", func(c *Config) { c.Sigma = math.NaN() }},
		{"zero repeats", func(c *Config) { c.Repeats = 0 }},
		{"too many repeats", func(c *Config) { c.Repeats = MaxRepeats + 1 }},
		{"zero scale cap", func(c *Config) { c.ScaleCap = 0 }},
		{"negative scale cap", func(c *Config) { c.ScaleCap = -2 }},
		{"negative fallback", func(c *Config) { c.FallbackScale = -1 }},
		{"fallback above cap", func(c *Config) { c.FallbackScale = 11 }},
		{"empty params", func(c *Config) { c.Layers[0].Params = nil }},
		{"NaN param", func(c *Config) { c.Layers[0].Params[0] = math.NaN() }},
		{"Inf update", func(c *Config) { c.Layers[0].Clients[0][0] = math.Inf(-1) }},
		{
			"shape mismatch within layer",
			func(c *Config) { c.Layers[0].Clients[1] = []float64{1} },
		},
		{
			"client count mismatch across layers",
			func(c *Config) {
				c.Layers = append(c.Layers, Layer{
					Name:    "second",
					Params:  []float64{1},
					Clients: [][]float64{{1}}, // only 1 of 2 clients
				})
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := validConfig()
			tc.mut(&c)
			if err := c.Validate(); err == nil {
				t.Fatalf("Validate accepted %s", tc.name)
			} else if _, err := Run(c); err == nil {
				t.Fatalf("Run accepted %s", tc.name)
			}
		})
	}
}

func TestRunRejectsAndKeepsNoReport(t *testing.T) {
	good := validConfig()
	rep, err := Run(good)
	if err != nil {
		t.Fatal(err)
	}
	// Independent call with bad config must return an error, not a report.
	bad := validConfig()
	bad.Budget = -5
	badRep, err := Run(bad)
	if err == nil {
		t.Fatal("negative budget accepted")
	}
	if badRep != nil {
		t.Fatal("report leaked for invalid config")
	}
	// The earlier good report is untouched (caller keeps it).
	if rep == nil || len(rep.Layers) != 1 {
		t.Fatal("previous report altered")
	}
}

func TestEmptyConfigIsSentinel(t *testing.T) {
	var c Config
	if !errors.Is(c.Validate(), ErrEmptyConfig) {
		t.Fatal("expected ErrEmptyConfig")
	}
}
