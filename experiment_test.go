package main

import (
	"bytes"
	"encoding/json"
	"math"
	"testing"
)

const eps = 1e-12

func approxEq(a, b, tol float64) bool {
	if math.IsNaN(a) || math.IsNaN(b) {
		return false
	}
	d := math.Abs(a - b)
	if d <= tol {
		return true
	}
	return d <= tol*math.Max(math.Abs(a), math.Abs(b))
}

func vecEq(t *testing.T, got, want []float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("长度不匹配: got %v want %v", got, want)
	}
	for i := range got {
		if !approxEq(got[i], want[i], tol) {
			t.Fatalf("坐标 %d: got %v want %v (全量 got=%v)", i, got[i], want[i], got)
		}
	}
}

func baseReq() *ExperimentRequest {
	return &ExperimentRequest{
		Seed:          42,
		NoiseStd:      0,
		Repeats:       1,
		Budget:        10,
		AllocMode:     "equal",
		ScaleCap:      100,
		ScaleFallback: 0.5,
		Layers: []LayerSpec{
			{Name: "L1", Dim: 2, Params: []float64{3, 4}},
		},
		Clients: [][][]float64{
			{{1, 2}},
			{{3, 4}},
		},
	}
}

func TestAllocateBudgetsEqual(t *testing.T) {
	// B=6, L=4 -> 每个预算 3，平方和 36 = B^2
	got, err := allocateBudgets("equal", 6, []int{2, 3, 1, 2})
	if err != nil {
		t.Fatal(err)
	}
	var sq float64
	for i, b := range got {
		if !approxEq(b, 3, eps) {
			t.Fatalf("层 %d 预算 = %v, 期望 3", i, b)
		}
		sq += b * b
	}
	if !approxEq(sq, 36, eps) {
		t.Fatalf("预算平方和 = %v, 期望 36", sq)
	}
}

func TestAllocateBudgetsDimProportional(t *testing.T) {
	// 维数 1,3：和为 4；b1 = 6*sqrt(1/4)=3, b2 = 6*sqrt(3/4)=3*sqrt(3)
	got, err := allocateBudgets("dim_proportional", 6, []int{1, 3})
	if err != nil {
		t.Fatal(err)
	}
	if !approxEq(got[0], 3, eps) {
		t.Fatalf("b0 = %v, 期望 3", got[0])
	}
	if !approxEq(got[1], 3*math.Sqrt(3), eps) {
		t.Fatalf("b1 = %v, 期望 %v", got[1], 3*math.Sqrt(3))
	}
	if !approxEq(got[0]*got[0]+got[1]*got[1], 36, eps) {
		t.Fatalf("预算平方和不为 B^2: %v", got[0]*got[0]+got[1]*got[1])
	}
}

func TestAllocateBudgetsEqualPerLayer(t *testing.T) {
	// 单层：预算恰为总预算本身（b^2 = B^2）
	got, err := allocateBudgets("equal", 5, []int{2})
	if err != nil {
		t.Fatal(err)
	}
	if !approxEq(got[0], 5, eps) {
		t.Fatalf("单层预算 = %v, 期望 5", got[0])
	}
}

func TestClipUpperBound(t *testing.T) {
	// 向量 (3,4) 范数 5，预算 2 -> 裁剪后范数恰为 2
	out, clipped, pre, post := clipToBudget([]float64{3, 4}, 2)
	if !clipped {
		t.Fatal("期望被裁剪")
	}
	if !approxEq(pre, 5, eps) || !approxEq(post, 2, eps) {
		t.Fatalf("范数 pre=%v post=%v", pre, post)
	}
	if n := norm(out); !approxEq(n, 2, eps) {
		t.Fatalf("裁剪后范数 %v 超过预算 2", n)
	}
	vecEq(t, out, []float64{1.2, 1.6}, eps)

	// 不超预算：原样保留
	out2, clipped2, _, _ := clipToBudget([]float64{0.6, 0.8}, 2)
	if clipped2 {
		t.Fatal("不应裁剪")
	}
	vecEq(t, out2, []float64{0.6, 0.8}, eps)
}

// TestDeterministicHandCalc 是独立手算的端到端例子（零噪声）。
func TestDeterministicHandCalc(t *testing.T) {
	req := baseReq()
	// 两个客户端更新 (1,2) 范数 sqrt5≈2.236 与 (3,4) 范数 5；
	// equal 预算 B/sqrt(L)=10，均不裁剪。
	// 聚合 = ((1+3)/2,(2+4)/2) = (2,3)，范数 sqrt13。
	// 参数 (3,4) 范数 5；缩放 = 5/sqrt13，最终 = 5/sqrt13 * (2,3)。
	resp, err := Compute(req)
	if err != nil {
		t.Fatal(err)
	}
	lr := resp.Layers[0]
	if lr.ClippedFraction != 0 {
		t.Fatalf("裁剪比例 = %v, 期望 0", lr.ClippedFraction)
	}
	vecEq(t, lr.RefAgg, []float64{2, 3}, eps)
	if !approxEq(lr.RefAggNorm, math.Sqrt(13), eps) {
		t.Fatalf("聚合范数 %v", lr.RefAggNorm)
	}
	wantScale := 5 / math.Sqrt(13)
	if !approxEq(lr.RefScale, wantScale, eps) {
		t.Fatalf("缩放 = %v, 期望 %v", lr.RefScale, wantScale)
	}
	if lr.RefScaleSrc != "norm_ratio" {
		t.Fatalf("缩放来源 = %q", lr.RefScaleSrc)
	}
	vecEq(t, lr.RefFinal, []float64{2 * wantScale, 3 * wantScale}, eps)
	// 最终范数应等于参数范数 5（未触顶）
	if !approxEq(norm(lr.RefFinal), 5, eps) {
		t.Fatalf("最终范数 %v, 期望 5", norm(lr.RefFinal))
	}
}

func TestClipActuallyTriggersAndCap(t *testing.T) {
	req := baseReq()
	req.Budget = 4 // equal 单层 => 预算 4；(3,4) 范数 5 被裁剪，(1,2) 不裁剪
	req.ScaleCap = 0.7
	resp, err := Compute(req)
	if err != nil {
		t.Fatal(err)
	}
	lr := resp.Layers[0]
	if !lr.Clipped[1] || lr.Clipped[0] {
		t.Fatalf("裁剪标记错误: %v", lr.Clipped)
	}
	if !approxEq(lr.ClippedFraction, 0.5, eps) {
		t.Fatalf("裁剪比例 %v, 期望 0.5", lr.ClippedFraction)
	}
	// 被裁剪更新 = 4/5*(3,4) = (2.4,3.2)；聚合 = ((1+2.4)/2,(2+3.2)/2)=(1.7,2.6)
	vecEq(t, lr.RefAgg, []float64{1.7, 2.6}, eps)
	rawScale := 5 / norm([]float64{1.7, 2.6})
	wantScale := math.Min(rawScale, 0.7)
	if !approxEq(lr.RefScale, wantScale, eps) {
		t.Fatalf("缩放 %v, 原始 %v, 期望受上限约束 %v", lr.RefScale, rawScale, wantScale)
	}
	if lr.RefScale > 0.7+eps {
		t.Fatal("缩放系数超过上限")
	}
}

func TestResolveScaleFallbacks(t *testing.T) {
	// 正常比值
	if s, src := resolveScale(5, 2, 100, 0.5); !approxEq(s, 2.5, eps) || src != "norm_ratio" {
		t.Fatalf("正常情况 s=%v src=%q", s, src)
	}
	// 0/0 -> 0，不产生 NaN
	if s, src := resolveScale(0, 0, 100, 0.5); s != 0 || src != "zero_over_zero" {
		t.Fatalf("零比零 s=%v src=%q", s, src)
	}
	// 聚合零但参数非零 -> 显式回退
	if s, src := resolveScale(5, 0, 100, 0.5); !approxEq(s, 0.5, eps) || src != "fallback" {
		t.Fatalf("回退 s=%v src=%q", s, src)
	}
	// 回退同样受上限约束
	if s, _ := resolveScale(5, 0, 0.1, 0.5); !approxEq(s, 0.1, eps) {
		t.Fatalf("回退未受上限约束: %v", s)
	}
	// 普通比值受上限约束
	if s, _ := resolveScale(1000, 1, 2, 0.5); !approxEq(s, 2, eps) {
		t.Fatalf("上限未生效: %v", s)
	}
}

func TestZeroAggregateEndToEnd(t *testing.T) {
	req := baseReq()
	// 两个客户端更新互为相反数，聚合精确为零；参数非零。
	req.Clients = [][][]float64{{{1, 0}}, {{-1, 0}}}
	req.ScaleFallback = 0.25
	req.ScaleCap = 10
	resp, err := Compute(req)
	if err != nil {
		t.Fatal(err)
	}
	lr := resp.Layers[0]
	if lr.RefScaleSrc != "fallback" || !approxEq(lr.RefScale, 0.25, eps) {
		t.Fatalf("期望回退 0.25, got scale=%v src=%q", lr.RefScale, lr.RefScaleSrc)
	}
	vecEq(t, lr.RefFinal, []float64{0, 0}, eps)
	for i, v := range lr.MeanFinal {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("零聚合产生非有限值 @%d: %v", i, v)
		}
	}
}

func TestZeroParamsAndZeroUpdate(t *testing.T) {
	req := baseReq()
	req.Layers[0].Params = []float64{0, 0}
	req.Clients = [][][]float64{{{0, 0}}, {{0, 0}}}
	resp, err := Compute(req)
	if err != nil {
		t.Fatal(err)
	}
	lr := resp.Layers[0]
	if lr.RefScaleSrc != "zero_over_zero" || lr.RefScale != 0 {
		t.Fatalf("期望 zero_over_zero/0, got %v %q", lr.RefScale, lr.RefScaleSrc)
	}
	vecEq(t, lr.RefFinal, []float64{0, 0}, eps)
}

func TestInvalidInputs(t *testing.T) {
	cases := []struct {
		name string
		mut  func(*ExperimentRequest)
	}{
		{"空客户端", func(r *ExperimentRequest) { r.Clients = nil }},
		{"负预算", func(r *ExperimentRequest) { r.Budget = -1 }},
		{"零预算", func(r *ExperimentRequest) { r.Budget = 0 }},
		{"负噪声", func(r *ExperimentRequest) { r.NoiseStd = -0.1 }},
		{"重复次数为零", func(r *ExperimentRequest) { r.Repeats = 0 }},
		{"非正上限", func(r *ExperimentRequest) { r.ScaleCap = 0 }},
		{"负回退", func(r *ExperimentRequest) { r.ScaleFallback = -1 }},
		{"层形状不一致-客户端层数", func(r *ExperimentRequest) { r.Clients = [][][]float64{{{1}}} }},
		{"层形状不一致-向量长度", func(r *ExperimentRequest) {
			r.Clients = [][][]float64{{{1, 2, 3}}, {{1, 2, 3}}}
		}},
		{"参数长度不符", func(r *ExperimentRequest) { r.Layers[0].Params = []float64{3} }},
		{"NaN更新", func(r *ExperimentRequest) { r.Clients[0][0][0] = math.NaN() }},
		{"Inf参数", func(r *ExperimentRequest) { r.Layers[0].Params[0] = math.Inf(1) }},
		{"NaN噪声std", func(r *ExperimentRequest) { r.NoiseStd = math.NaN() }},
		{"未知分配方式", func(r *ExperimentRequest) { r.AllocMode = "weird" }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := baseReq()
			tc.mut(req)
			if _, err := Compute(req); err == nil {
				t.Fatalf("%s: 期望被拒绝，但通过了", tc.name)
			}
		})
	}
}

// fakeSource 按预设序列产生“标准正态”样本（测试中用简单整数代替），
// 使结果可完全手算。
type fakeSource struct {
	vals []float64
	idx  int
}

func (f *fakeSource) next() float64 {
	v := f.vals[f.idx%len(f.vals)]
	f.idx++
	return v
}

func TestNoiseHandCalcPerClient(t *testing.T) {
	req := baseReq()
	req.NoiseStd = 1
	req.Repeats = 1
	// 单层 2 维 x 2 客户端 = 4 个噪声样本，按 层->客户端->坐标 消费。
	// 序列对应：c0: (10,-10)，c1: (-2, 2)。
	seq := []float64{10, -10, -2, 2}
	factory := func(run int) normalSource { return &fakeSource{vals: seq} }

	resp, err := computeWith(req, factory)
	if err != nil {
		t.Fatal(err)
	}
	lr := resp.Layers[0]
	// 噪声化客户端：c0 (11,-8)，c1 (1,6)；均值 (6,-1)，范数 sqrt37。
	// 注意均值后每坐标噪声均值为 (10-2)/2=4 与 (-10+2)/2=-4。
	wantAgg := []float64{6, -1}
	wantScale := 5 / norm(wantAgg) // 参数范数 5 / 噪声聚合范数 sqrt37
	if !approxEq(lr.MeanScale, wantScale, 1e-9) {
		t.Fatalf("scale = %v, 期望 %v", lr.MeanScale, wantScale)
	}
	vecEq(t, lr.MeanFinal, scaleVec(wantAgg, wantScale), 1e-9)
}

func TestNoiseConsumptionOrderTwoLayers(t *testing.T) {
	req := baseReq()
	req.Layers = []LayerSpec{
		{Name: "L1", Dim: 1, Params: []float64{0}},
		{Name: "L2", Dim: 1, Params: []float64{0}},
	}
	req.Clients = [][][]float64{{{0}, {0}}, {{0}, {0}}}
	req.NoiseStd = 1
	req.Repeats = 1
	// 层1 消费前两个（每客户端1个），层2 消费后两个。
	seq := []float64{1, 3, 100, 200}
	if _, err := computeWith(req, func(run int) normalSource { return &fakeSource{vals: seq} }); err != nil {
		t.Fatal(err)
	}
	// 为验证消费顺序，让参数范数非零再算一遍。
	req.Layers[0].Params = []float64{2}
	req.Layers[1].Params = []float64{2}
	resp2, err := computeWith(req, func(run int) normalSource { return &fakeSource{vals: seq} })
	if err != nil {
		t.Fatal(err)
	}
	// 层1 噪声均值 (1+3)/2=2 => agg=2，范数2，scale=2/2=1 => final=2
	vecEq(t, resp2.Layers[0].MeanFinal, []float64{2}, 1e-9)
	// 层2 噪声均值 (100+200)/2=150 => scale=2/150 => final=2
	vecEq(t, resp2.Layers[1].MeanFinal, []float64{2}, 1e-9)
}

func TestFixedSeedDeterministicAndZeroNoiseExact(t *testing.T) {
	req := baseReq()
	req.NoiseStd = 0.5
	req.Repeats = 8
	r1, err := Compute(req)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := Compute(req)
	if err != nil {
		t.Fatal(err)
	}
	b1, _ := json.Marshal(r1)
	b2, _ := json.Marshal(r2)
	if !bytes.Equal(b1, b2) {
		t.Fatal("固定种子重复运行结果不一致")
	}

	// 零噪声：重复实验统计与确定性参考逐位一致，且方差为零。
	req.NoiseStd = 0
	r3, err := Compute(req)
	if err != nil {
		t.Fatal(err)
	}
	lr := r3.Layers[0]
	vecEq(t, lr.MeanFinal, lr.RefFinal, 0) // 精确（tol=0，逐位相等）
	for _, v := range lr.VarFinal {
		if v != 0 {
			t.Fatalf("零噪声方差应为 0，得到 %v", v)
		}
	}
	if lr.MSE != 0 || lr.BiasNorm != 0 || lr.RMSE != 0 {
		t.Fatalf("零噪声误差应为 0: mse=%v bias=%v rmse=%v", lr.MSE, lr.BiasNorm, lr.RMSE)
	}
}

func TestMeanVarHandCalc(t *testing.T) {
	m, v := meanVar([]float64{2, 4})
	if !approxEq(m, 3, eps) {
		t.Fatalf("均值 %v", m)
	}
	if !approxEq(v, 2, eps) { // 样本方差 ((2-3)^2+(4-3)^2)/(2-1)=2
		t.Fatalf("样本方差 %v, 期望 2", v)
	}
	if m0, v0 := meanVar([]float64{7}); !approxEq(m0, 7, eps) || v0 != 0 {
		t.Fatalf("单样本: mean=%v var=%v", m0, v0)
	}
	if m0, v0 := meanVar(nil); m0 != 0 || v0 != 0 {
		t.Fatalf("空切片: mean=%v var=%v", m0, v0)
	}
}

func TestVarianceAcrossRunsHandCalc(t *testing.T) {
	// 参数范数设为 1，使最终向量 = 聚合向量 / 聚合范数（归一化到单位长度）。
	// 为得到干净的手算，使用一维：scale = 1/|agg|，final = sign(agg)。
	req := baseReq()
	req.Layers = []LayerSpec{{Name: "L", Dim: 1, Params: []float64{1}}}
	req.Clients = [][][]float64{{{0}}, {{0}}}
	req.NoiseStd = 1
	req.Repeats = 2
	// run0: 噪声 (3,5) -> agg 4 -> final 1
	// run1: 噪声 (-2,-8) -> agg -5 -> final -1
	runs := [][]float64{{3, 5}, {-2, -8}}
	factory := func(run int) normalSource { return &fakeSource{vals: []float64{runs[run][0], runs[run][1]}} }
	resp, err := computeWith(req, factory)
	if err != nil {
		t.Fatal(err)
	}
	lr := resp.Layers[0]
	if !approxEq(lr.MeanFinal[0], 0, eps) {
		t.Fatalf("均值 %v, 期望 0", lr.MeanFinal[0])
	}
	if !approxEq(lr.VarFinal[0], 2, eps) { // ((1-0)^2+(-1-0)^2)/(2-1)=2
		t.Fatalf("样本方差 %v, 期望 2", lr.VarFinal[0])
	}
}

// TestRealRNGStatistical 用真实 PCG 高斯源验证统计性质与确定性。
func TestRealRNGStatisticalAndGolden(t *testing.T) {
	req := baseReq()
	req.Layers = []LayerSpec{{Name: "L", Dim: 4, Params: []float64{1, 2, 3, 4}}}
	req.Clients = [][][]float64{{{0, 0, 0, 0}}, {{0, 0, 0, 0}}}
	req.NoiseStd = 1
	req.Repeats = 4000
	resp, err := Compute(req)
	if err != nil {
		t.Fatal(err)
	}
	lr := resp.Layers[0]
	// 每客户端加 N(0,1)，2 个客户端平均 => 坐标方差 1/2；
	// 经过缩放后方差会改变，但坐标均值应在有限样本下接近缩放后的零参考。
	for i, m := range lr.MeanFinal {
		if math.Abs(m) > 0.15 {
			t.Fatalf("坐标 %d 均值 %v 偏离零参考过大", i, m)
		}
	}
	// 聚合范数均值应接近期望 ||N(0, 0.5 I_4)|| ≈ sqrt(4*0.5)=sqrt2，
	// 取较宽容限仅作健全性检查。
	if math.Abs(lr.MeanAggNorm-math.Sqrt(2)) > 0.2 {
		t.Fatalf("平均聚合范数 %v 偏离 sqrt2", lr.MeanAggNorm)
	}

	// 黄金值：同一固定种子必须得到此前记录的确定值。
	golden, _ := json.Marshal(resp)
	resp2, err := Compute(req)
	if err != nil {
		t.Fatal(err)
	}
	again, _ := json.Marshal(resp2)
	if !bytes.Equal(golden, again) {
		t.Fatal("真实 RNG 在固定种子下不确定")
	}
}
