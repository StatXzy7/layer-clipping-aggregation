package main

import (
	"fmt"
	"math"
	"math/rand/v2"
)

// ExperimentRequest 描述一次完整的数值实验。
type ExperimentRequest struct {
	Seed          uint64        `json:"seed"`
	NoiseStd      float64       `json:"noise_std"`
	Repeats       int           `json:"repeats"`
	Budget        float64       `json:"budget"`
	AllocMode     string        `json:"alloc_mode"` // "equal" 或 "dim_proportional"
	ScaleCap      float64       `json:"scale_cap"`
	ScaleFallback float64       `json:"scale_fallback"`
	Layers        []LayerSpec   `json:"layers"`
	Clients       [][][]float64 `json:"clients"`
}

// LayerSpec 是一个网络层的参数向量与维数。
type LayerSpec struct {
	Name   string    `json:"name"`
	Dim    int       `json:"dim"`
	Params []float64 `json:"params"`
}

// LayerResult 保存某一层在参考与重复实验下的全部数值。
type LayerResult struct {
	Index     int     `json:"index"`
	Name      string  `json:"name"`
	Dim       int     `json:"dim"`
	Budget    float64 `json:"budget"`
	ParamNorm float64 `json:"param_norm"`

	PreNorms        []float64 `json:"pre_norms"`
	PostNorms       []float64 `json:"post_norms"`
	Clipped         []bool    `json:"clipped"`
	ClippedFraction float64   `json:"clipped_fraction"`

	RefAgg      []float64 `json:"ref_agg"`
	RefAggNorm  float64   `json:"ref_agg_norm"`
	RefScale    float64   `json:"ref_scale"`
	RefScaleSrc string    `json:"ref_scale_source"`
	RefFinal    []float64 `json:"ref_final"`

	MeanFinal   []float64 `json:"mean_final"`
	VarFinal    []float64 `json:"var_final"`
	MeanAggNorm float64   `json:"mean_agg_norm"`
	VarAggNorm  float64   `json:"var_agg_norm"`
	MeanScale   float64   `json:"mean_scale"`
	VarScale    float64   `json:"var_scale"`

	BiasNorm float64 `json:"bias_norm"`
	MSE      float64 `json:"mse"`
	RMSE     float64 `json:"rmse"`
	RelError float64 `json:"rel_error"`
}

// ExperimentResponse 是计算结果。
type ExperimentResponse struct {
	Budgets []float64     `json:"budgets"`
	Layers  []LayerResult `json:"layers"`
	Note    string        `json:"note"`
}

// validate 检查所有输入。返回错误时不产生任何部分结果，
// 上层会保留上次有效显示。
func (r *ExperimentRequest) validate() error {
	if r.Repeats < 1 {
		return fmt.Errorf("重复次数必须为正整数，当前为 %d", r.Repeats)
	}
	if r.NoiseStd < 0 || !isFinite(r.NoiseStd) {
		return fmt.Errorf("噪声标准差必须为非负有限值，当前为 %v", r.NoiseStd)
	}
	if r.Budget <= 0 || !isFinite(r.Budget) {
		return fmt.Errorf("总裁剪预算必须为正有限值，当前为 %v", r.Budget)
	}
	if r.ScaleCap <= 0 || !isFinite(r.ScaleCap) {
		return fmt.Errorf("缩放系数上限必须为正有限值，当前为 %v", r.ScaleCap)
	}
	if r.ScaleFallback < 0 || !isFinite(r.ScaleFallback) {
		return fmt.Errorf("零范数回退系数必须为非负有限值，当前为 %v", r.ScaleFallback)
	}
	if r.AllocMode != "equal" && r.AllocMode != "dim_proportional" {
		return fmt.Errorf("未知预算分配方式 %q，应为 equal 或 dim_proportional", r.AllocMode)
	}
	if len(r.Layers) == 0 {
		return fmt.Errorf("至少需要一个层")
	}
	if len(r.Clients) == 0 {
		return fmt.Errorf("至少需要一个客户端：空客户端集合被拒绝")
	}
	for l, layer := range r.Layers {
		if layer.Dim <= 0 {
			return fmt.Errorf("第 %d 层维数必须为正，当前为 %d", l+1, layer.Dim)
		}
		if len(layer.Params) != layer.Dim {
			return fmt.Errorf("第 %d 层参数长度 %d 与维数 %d 不一致",
				l+1, len(layer.Params), layer.Dim)
		}
		for i, v := range layer.Params {
			if !isFinite(v) {
				return fmt.Errorf("第 %d 层第 %d 个参数不是有限数值", l+1, i+1)
			}
		}
	}
	for c, client := range r.Clients {
		if len(client) != len(r.Layers) {
			return fmt.Errorf("客户端 %d 提供 %d 层更新，但模型有 %d 层（层形状不一致）",
				c+1, len(client), len(r.Layers))
		}
		for l, update := range client {
			if len(update) != r.Layers[l].Dim {
				return fmt.Errorf("客户端 %d 第 %d 层更新长度 %d 与层维数 %d 不一致",
					c+1, l+1, len(update), r.Layers[l].Dim)
			}
			for i, v := range update {
				if !isFinite(v) {
					return fmt.Errorf("客户端 %d 第 %d 层第 %d 个更新不是有限数值",
						c+1, l+1, i+1)
				}
			}
		}
	}
	return nil
}

func isFinite(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}

// allocateBudgets 返回每层的 L2 裁剪预算。
//
//   - equal: 总裁剪预算按层数均分，b_i = B/sqrt(L)，
//     因而 sum b_i^2 = B^2。
//   - dim_proportional: 层预算平方与层维数成比例，
//     b_i = B*sqrt(d_i / sum_j d_j)，同样有 sum b_i^2 = B^2。
func allocateBudgets(mode string, budget float64, dims []int) ([]float64, error) {
	L := len(dims)
	out := make([]float64, L)
	switch mode {
	case "equal":
		b := budget / math.Sqrt(float64(L))
		for i := range out {
			out[i] = b
		}
		return out, nil
	case "dim_proportional":
		sumDim := 0
		for _, d := range dims {
			sumDim += d
		}
		if sumDim <= 0 {
			return nil, fmt.Errorf("层维数之和必须为正")
		}
		for i, d := range dims {
			out[i] = budget * math.Sqrt(float64(d)/float64(sumDim))
		}
		return out, nil
	default:
		return nil, fmt.Errorf("未知预算分配方式 %q", mode)
	}
}

func norm(v []float64) float64 {
	var s float64
	for _, x := range v {
		s += x * x
	}
	return math.Sqrt(s)
}

func cloneVec(v []float64) []float64 {
	out := make([]float64, len(v))
	copy(out, v)
	return out
}

// clipToBudget 按欧氏范数裁剪。范数不超过预算时原样返回（返回的是副本）。
func clipToBudget(v []float64, budget float64) (out []float64, clipped bool, pre, post float64) {
	pre = norm(v)
	out = cloneVec(v)
	if pre > budget {
		clipped = true
		scale := budget / pre
		for i := range out {
			out[i] *= scale
		}
		post = budget
	} else {
		post = pre
	}
	return out, clipped, pre, post
}

// resolveScale 计算层级缩放系数 ||params|| / ||agg||，
// 并处理聚合更新范数为零的情况：
//   - 参数范数也为 0：系数 0（0/0 定义为 0，结果保持零向量）；
//   - 参数范数 > 0：使用显式有限回退 scaleFallback。
//
// 两种情况随后都施加同一正系数上限 scaleCap。
func resolveScale(paramNorm, aggNorm, cap_, fallback float64) (scale float64, source string) {
	switch {
	case aggNorm > 0:
		scale, source = paramNorm/aggNorm, "norm_ratio"
	case paramNorm == 0:
		scale, source = 0, "zero_over_zero"
	default:
		scale, source = fallback, "fallback"
	}
	if scale > cap_ {
		scale = cap_
	}
	if !isFinite(scale) {
		scale = 0
	}
	return scale, source
}

// normalSource 产生标准正态 N(0,1) 样本。噪声样本在每个重复内部按
// “层 -> 客户端 -> 坐标”的顺序消费；sigma == 0 时不消费任何样本，
// 因此零噪声路径与确定性参考逐位一致。
type normalSource interface {
	next() float64
}

// rngSource 封装 math/rand/v2 的 PCG 生成器。
type rngSource struct{ r *rand.Rand }

func (s *rngSource) next() float64 { return s.r.NormFloat64() }

// newRNG 用固定种子构造生成器。流号参与混合，保证不同重复互不相同，
// 而同一固定种子始终得到同一序列（重复运行一致）。
func newRNG(seed uint64, stream uint64) *rand.Rand {
	return rand.New(rand.NewPCG(mix64(seed, 0x9E3779B97F4A7C15), mix64(stream^seed, 0xD1B54A32D192ED03)))
}

func mix64(a, b uint64) uint64 {
	x := a + b + 0x9E3779B97F4A7C15
	x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
	x = (x ^ (x >> 27)) * 0x94D049BB133111EB
	return x ^ (x >> 31)
}

// Compute 运行整个实验（使用真实 PCG 高斯噪声源）。任何输入错误都会在计算前被拒绝。
func Compute(req *ExperimentRequest) (*ExperimentResponse, error) {
	return computeWith(req, nil)
}

// sourceFactory 在每个重复开始时产生一个噪声源；返回 nil 表示零噪声。
// factory 本身为 nil 时使用由固定种子派生的默认源。
type sourceFactory func(run int) normalSource

func computeWith(req *ExperimentRequest, factory sourceFactory) (*ExperimentResponse, error) {
	if err := req.validate(); err != nil {
		return nil, err
	}

	L := len(req.Layers)
	dims := make([]int, L)
	for i, layer := range req.Layers {
		dims[i] = layer.Dim
	}
	budgets, err := allocateBudgets(req.AllocMode, req.Budget, dims)
	if err != nil {
		return nil, err
	}

	C := len(req.Clients)
	invC := 1.0 / float64(C)
	results := make([]LayerResult, L)

	// clipped[l][c] 存放客户端 c 在层 l 裁剪后的更新。
	clipped := make([][][]float64, L)
	for l := 0; l < L; l++ {
		res := &results[l]
		res.Index = l
		res.Name = req.Layers[l].Name
		res.Dim = dims[l]
		res.Budget = budgets[l]
		res.ParamNorm = norm(req.Layers[l].Params)
		res.PreNorms = make([]float64, C)
		res.PostNorms = make([]float64, C)
		res.Clipped = make([]bool, C)
		clipped[l] = make([][]float64, C)

		nClipped := 0
		for c := 0; c < C; c++ {
			out, was, pre, post := clipToBudget(req.Clients[c][l], budgets[l])
			clipped[l][c] = out
			res.Clipped[c] = was
			res.PreNorms[c] = pre
			res.PostNorms[c] = post
			if was {
				nClipped++
			}
		}
		res.ClippedFraction = float64(nClipped) / float64(C)

		// 无噪声参考：等权聚合裁剪后更新。
		refAgg := make([]float64, dims[l])
		for c := 0; c < C; c++ {
			for i, v := range clipped[l][c] {
				refAgg[i] += v
			}
		}
		for i := range refAgg {
			refAgg[i] *= invC
		}
		res.RefAgg = refAgg
		res.RefAggNorm = norm(refAgg)
		res.RefScale, res.RefScaleSrc = resolveScale(
			res.ParamNorm, res.RefAggNorm, req.ScaleCap, req.ScaleFallback)
		res.RefFinal = scaleVec(refAgg, res.RefScale)
	}

	// 每个重复：对每个客户端的每个坐标加独立高斯噪声，
	// 再等权聚合，最后做层级缩放。
	R := req.Repeats
	finals := make([][][]float64, L)
	aggNorms := make([][]float64, L)
	scales := make([][]float64, L)
	for l := range finals {
		finals[l] = make([][]float64, R)
		aggNorms[l] = make([]float64, R)
		scales[l] = make([]float64, R)
	}

	defaultFactory := func(run int) normalSource {
		if req.NoiseStd <= 0 {
			return nil
		}
		return &rngSource{r: newRNG(req.Seed, uint64(run)+1)}
	}
	for r := 0; r < R; r++ {
		var src normalSource
		if factory != nil {
			src = factory(r)
		} else {
			src = defaultFactory(r)
		}
		for l := 0; l < L; l++ {
			d := dims[l]
			agg := make([]float64, d)
			for c := 0; c < C; c++ {
				for i := 0; i < d; i++ {
					v := clipped[l][c][i]
					if src != nil {
						v += req.NoiseStd * src.next()
					}
					agg[i] += v
				}
			}
			for i := range agg {
				agg[i] *= invC
			}
			s, _ := resolveScale(results[l].ParamNorm, norm(agg), req.ScaleCap, req.ScaleFallback)
			aggNorms[l][r] = norm(agg)
			scales[l][r] = s
			finals[l][r] = scaleVec(agg, s)
		}
	}

	for l := 0; l < L; l++ {
		res := &results[l]
		d := dims[l]
		res.MeanFinal = make([]float64, d)
		res.VarFinal = make([]float64, d)
		for r := 0; r < R; r++ {
			for i := 0; i < d; i++ {
				res.MeanFinal[i] += finals[l][r][i]
			}
		}
		for i := 0; i < d; i++ {
			res.MeanFinal[i] /= float64(R)
		}
		if R > 1 {
			for r := 0; r < R; r++ {
				for i := 0; i < d; i++ {
					diff := finals[l][r][i] - res.MeanFinal[i]
					res.VarFinal[i] += diff * diff
				}
			}
			denom := float64(R - 1)
			for i := range res.VarFinal {
				res.VarFinal[i] /= denom
			}
		}

		res.MeanAggNorm, res.VarAggNorm = meanVar(aggNorms[l])
		res.MeanScale, res.VarScale = meanVar(scales[l])

		// 与无噪声参考比较：偏差范数、MSE、RMSE、相对误差。
		var biasSq, mse float64
		for i := 0; i < d; i++ {
			bias := res.MeanFinal[i] - res.RefFinal[i]
			biasSq += bias * bias
			for r := 0; r < R; r++ {
				e := finals[l][r][i] - res.RefFinal[i]
				mse += e * e
			}
		}
		res.BiasNorm = math.Sqrt(biasSq)
		mse /= float64(R * d)
		res.MSE = mse
		res.RMSE = math.Sqrt(mse)
		refNorm := norm(res.RefFinal)
		if refNorm > 0 {
			res.RelError = res.BiasNorm / refNorm
		} else {
			res.RelError = res.BiasNorm
		}
	}

	return &ExperimentResponse{
		Budgets: budgets,
		Layers:  results,
		Note: fmt.Sprintf("已完成 %d 个客户端 x %d 层 x %d 次重复（预算分配：%s）",
			C, L, R, req.AllocMode),
	}, nil
}

func scaleVec(v []float64, s float64) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = x * s
	}
	return out
}

// meanVar 返回均值与样本方差（自由度 R-1；R==1 时方差为 0）。
func meanVar(xs []float64) (mean, variance float64) {
	if len(xs) == 0 {
		return 0, 0
	}
	for _, x := range xs {
		mean += x
	}
	mean /= float64(len(xs))
	if len(xs) > 1 {
		for _, x := range xs {
			d := x - mean
			variance += d * d
		}
		variance /= float64(len(xs) - 1)
	}
	return mean, variance
}
