package core

import (
	"math"
	"testing"
)

func approxEq(t *testing.T, got, want, tol float64, msg string) {
	t.Helper()
	if math.IsNaN(got) || math.Abs(got-want) > tol {
		t.Fatalf("%s: got %.17g want %.17g", msg, got, want)
	}
}

func TestAllocateBudgetsEqual(t *testing.T) {
	// b = B/sqrt(L): sum b^2 == B^2 exactly (in exact arithmetic).
	b := AllocateBudgets(4, []int{2, 3, 5, 1}, BudgetEqual)
	if len(b) != 4 {
		t.Fatalf("got %d budgets", len(b))
	}
	want := 4 / math.Sqrt(4) // 2
	var sumSq float64
	for i, x := range b {
		approxEq(t, x, want, 1e-15, "equal budget")
		sumSq += x * x
		if i > 0 && x != b[0] {
			t.Fatalf("budgets not equal: %v", b)
		}
	}
	approxEq(t, sumSq, 16, 1e-14, "sum of squares equals B^2")

	// L=2 hand check: B=4 -> each 2*sqrt(2).
	b2 := AllocateBudgets(4, []int{2, 3}, BudgetEqual)
	approxEq(t, b2[0], 2*math.Sqrt2, 1e-15, "b[0]")
	approxEq(t, b2[1], 2*math.Sqrt2, 1e-15, "b[1]")
	approxEq(t, b2[0]*b2[0]+b2[1]*b2[1], 16, 1e-14, "sum sq L=2")
}

func TestAllocateBudgetsProportional(t *testing.T) {
	// dims 1 and 3, B=4: b0=4*sqrt(1/4)=2, b1=4*sqrt(3/4)=2*sqrt(3).
	b := AllocateBudgets(4, []int{1, 3}, BudgetProportional)
	approxEq(t, b[0], 2, 1e-15, "b0")
	approxEq(t, b[1], 2*math.Sqrt(3), 1e-15, "b1")
	approxEq(t, b[0]*b[0]+b[1]*b[1], 16, 1e-14, "sum of squares equals B^2")

	// Uneven dims 1,1,2, B=3: b = 1.5, 1.5, 1.5*sqrt(2).
	b2 := AllocateBudgets(3, []int{1, 1, 2}, BudgetProportional)
	approxEq(t, b2[0], 1.5, 1e-15, "b2[0]")
	approxEq(t, b2[1], 1.5, 1e-15, "b2[1]")
	approxEq(t, b2[2], 1.5*math.Sqrt2, 1e-15, "b2[2]")
	var sq float64
	for _, x := range b2 {
		sq += x * x
	}
	approxEq(t, sq, 9, 1e-14, "sum sq uneven")
}

func TestClipByNorm(t *testing.T) {
	// Norm 5 (3,4) within budget 5: untouched.
	v := []float64{3, 4}
	before, after, clipped, ratio := ClipByNorm(v, 5)
	approxEq(t, before, 5, 1e-15, "before")
	approxEq(t, after, 5, 1e-15, "after")
	if clipped || ratio != 0 || v[0] != 3 || v[1] != 4 {
		t.Fatalf("unexpected clip: %v %v %v", clipped, ratio, v)
	}

	// Norm 6 (3/5*6, 4/5*6) clipped to budget 3: half of each component.
	v = []float64{3.6, 4.8}
	before, after, clipped, ratio = ClipByNorm(v, 3)
	approxEq(t, before, 6, 1e-15, "before6")
	approxEq(t, after, 3, 1e-15, "after3")
	if !clipped {
		t.Fatal("expected clipped")
	}
	approxEq(t, ratio, 0.5, 1e-15, "removed fraction")
	approxEq(t, v[0], 1.8, 1e-15, "v0")
	approxEq(t, v[1], 2.4, 1e-15, "v1")

	// (1,2,2) norm 3 clipped to budget 2.
	v = []float64{1, 2, 2}
	_, _, clipped, ratio = ClipByNorm(v, 2)
	if !clipped {
		t.Fatal("expected clipped")
	}
	approxEq(t, ratio, 1.0/3.0, 1e-15, "ratio 1/3")
	approxEq(t, v[0], 2.0/3.0, 1e-15, "v0=2/3")
	approxEq(t, v[1], 4.0/3.0, 1e-15, "v1=4/3")
	approxEq(t, v[2], 4.0/3.0, 1e-15, "v2=4/3")

	// Zero vector: norm 0, never clipped, no division by zero.
	v = []float64{0, 0}
	before, after, clipped, ratio = ClipByNorm(v, 3)
	approxEq(t, before, 0, 0, "zero before")
	approxEq(t, after, 0, 0, "zero after")
	if clipped || ratio != 0 {
		t.Fatalf("zero vector clipped: %v %v", clipped, ratio)
	}
}

func TestScaleCoeff(t *testing.T) {
	// 10/2.5 = 4, below cap 10.
	approxEq(t, ScaleCoeff(10, 2.5, 10, 1), 4, 1e-15, "plain ratio")
	// Cap binds: 10/0.5 = 20 -> 3.
	approxEq(t, ScaleCoeff(10, 0.5, 3, 1), 3, 0, "cap")
	// Zero aggregate norm -> explicit finite fallback, no NaN/Inf.
	got := ScaleCoeff(10, 0, 3, 0.5)
	approxEq(t, got, 0.5, 0, "fallback with nonzero params")
	got = ScaleCoeff(0, 0, 3, 0.5)
	approxEq(t, got, 0.5, 0, "fallback with zero params")
	if math.IsNaN(got) || math.IsInf(got, 0) {
		t.Fatal("fallback not finite")
	}
}
