package main

import (
	"bytes"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestComputeHandlerOK(t *testing.T) {
	req := baseReq()
	body, _ := json.Marshal(req)
	rec := httptest.NewRecorder()
	hreq := httptest.NewRequest(http.MethodPost, "/api/compute", bytes.NewReader(body))
	computeHandler(rec, hreq)
	if rec.Code != http.StatusOK {
		t.Fatalf("状态码 %d, body=%s", rec.Code, rec.Body.String())
	}
	var resp ExperimentResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Layers) != 1 {
		t.Fatalf("层数 %d", len(resp.Layers))
	}
}

func TestComputeHandlerRejectsInvalid(t *testing.T) {
	bad := []byte(`{"seed":1,"noise_std":-1,"repeats":1,"budget":10,"alloc_mode":"equal","scale_cap":100,"scale_fallback":0.5,"layers":[],"clients":[]}`)
	rec := httptest.NewRecorder()
	hreq := httptest.NewRequest(http.MethodPost, "/api/compute", bytes.NewReader(bad))
	computeHandler(rec, hreq)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("非法输入应 400，得到 %d", rec.Code)
	}
	var eb errBody
	if err := json.Unmarshal(rec.Body.Bytes(), &eb); err != nil || eb.Error == "" {
		t.Fatalf("期望错误消息, got %s", rec.Body.String())
	}
}

func TestComputeHandlerBadJSON(t *testing.T) {
	rec := httptest.NewRecorder()
	hreq := httptest.NewRequest(http.MethodPost, "/api/compute", strings.NewReader("{not json"))
	computeHandler(rec, hreq)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("坏 JSON 应 400，得到 %d", rec.Code)
	}
}

func TestIndexServed(t *testing.T) {
	rec := httptest.NewRecorder()
	hreq := httptest.NewRequest(http.MethodGet, "/", nil)
	indexHandler(rec, hreq)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "数值实验台") {
		t.Fatalf("页面未正确提供: code=%d", rec.Code)
	}
}

// TestPortZeroPrintsAddress 构建当前二进制并用 --port 0 启动，
// 断言它打印了实际地址且该端口可连接，随后结束进程。
func TestPortZeroPrintsAddress(t *testing.T) {
	if testing.Short() {
		t.Skip("跳过需要构建子进程的测试")
	}
	bin := "./layerclip_test_bin"
	t.Cleanup(func() { _ = os.Remove(bin) })
	if build := exec.Command("go", "build", "-o", bin, "."); build.Run() != nil {
		t.Fatal("构建测试二进制失败")
	}
	cmd := exec.Command(bin, "--host", "127.0.0.1", "--port", "0")
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cmd.Process.Kill(); _ = cmd.Wait() }()

	got := ""
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if strings.Contains(out.String(), "http://") {
			got = out.String()
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if got == "" {
		t.Fatalf("未打印实际地址: %q", out.String())
	}
	idx := strings.Index(got, "http://127.0.0.1:")
	rest := got[idx+len("http://127.0.0.1:"):]
	portStr := rest
	if nl := strings.IndexAny(portStr, "\n "); nl >= 0 {
		portStr = portStr[:nl]
	}
	port, err := strconv.Atoi(portStr)
	if err != nil || port == 0 {
		t.Fatalf("未解析到非零实际端口: %q (err=%v)", portStr, err)
	}
	conn, err := net.DialTimeout("tcp", "127.0.0.1:"+portStr, time.Second)
	if err != nil {
		t.Fatalf("实际端口 %s 无法连接: %v", portStr, err)
	}
	_ = conn.Close()
}
