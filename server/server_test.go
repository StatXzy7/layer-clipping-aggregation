package server

import (
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"layerbench/core"
)

func newTestServer(t *testing.T) *Server {
	t.Helper()
	return New([]byte("<html>ok</html>"))
}

const goodBody = `{
  "layers":[{"name":"L0","params":[6,8],"clients":[[3,4],[0,0]]}],
  "budget":5,"budgetMode":"equal","sigma":0.5,"seed":42,
  "repeats":2,"scaleCap":10,"fallbackScale":1
}`

func TestRunAndLast(t *testing.T) {
	s := newTestServer(t)

	// Initially there is no report.
	req := httptest.NewRequest(http.MethodGet, "/api/last", nil)
	rec := httptest.NewRecorder()
	s.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("last code = %d", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), `"report":null`) {
		t.Fatalf("expected empty last, got %s", rec.Body.String())
	}

	// Valid run.
	req = httptest.NewRequest(http.MethodPost, "/api/run", strings.NewReader(goodBody))
	req.Header.Set("Content-Type", "application/json")
	rec = httptest.NewRecorder()
	s.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("run code = %d body=%s", rec.Code, rec.Body.String())
	}
	var rep core.Report
	if err := json.Unmarshal(rec.Body.Bytes(), &rep); err != nil {
		t.Fatal(err)
	}
	if rep.N != 2 || len(rep.Layers) != 1 {
		t.Fatalf("unexpected report: %+v", rep)
	}
	if got := rep.Layers[0].RefUpdate[0]; got != 6 {
		t.Fatalf("refUpdate x = %v", got)
	}

	// /api/last now serves the stored report.
	req = httptest.NewRequest(http.MethodGet, "/api/last", nil)
	rec = httptest.NewRecorder()
	s.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"numClients":2`) {
		t.Fatalf("last did not serve stored report: %d %s", rec.Code, rec.Body.String())
	}
}

func TestInvalidRunRejectedAndLastKept(t *testing.T) {
	s := newTestServer(t)

	run := func(body string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodPost, "/api/run", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		s.ServeHTTP(rec, req)
		return rec
	}

	if rec := run(goodBody); rec.Code != http.StatusOK {
		t.Fatalf("good run failed: %s", rec.Body.String())
	}
	want := s.Last()

	badBodies := map[string]string{
		"negative budget": `{"layers":[{"name":"x","params":[1],"clients":[[1]]}],"budget":-1,"budgetMode":"equal","repeats":1,"scaleCap":10,"fallbackScale":1}`,
		"shape mismatch":  `{"layers":[{"name":"x","params":[1,2],"clients":[[1]]}],"budget":1,"budgetMode":"equal","repeats":1,"scaleCap":10,"fallbackScale":1}`,
		"empty clients":   `{"layers":[{"name":"x","params":[1],"clients":[]}],"budget":1,"budgetMode":"equal","repeats":1,"scaleCap":10,"fallbackScale":1}`,
		"NaN via token":   `{"layers":[{"name":"x","params":[NaN],"clients":[[1]]}],"budget":1,"budgetMode":"equal","repeats":1,"scaleCap":10,"fallbackScale":1}`,
		"malformed json":  `{not json`,
	}
	for name, body := range badBodies {
		rec := run(body)
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("%s: code=%d body=%s", name, rec.Code, rec.Body.String())
		}
		if !strings.Contains(rec.Body.String(), "error") {
			t.Fatalf("%s: missing error field: %s", name, rec.Body.String())
		}
	}

	// Stored report unchanged.
	if got := s.Last(); got != want {
		t.Fatal("last valid report was replaced by a rejected submission")
	}

	// The decoder accepts the bare NaN token into float64, so validation
	// (covered separately) must reject it with 400.
	body := `{"layers":[{"name":"x","params":[1],"clients":[[NaN]]}],"budget":1,"budgetMode":"equal","repeats":1,"scaleCap":10,"fallbackScale":1}`
	if rec := run(body); rec.Code != http.StatusBadRequest {
		t.Fatalf("NaN token body accepted: %d %s", rec.Code, rec.Body.String())
	}
}

func TestIndexServed(t *testing.T) {
	s := New([]byte("<html>page</html>"))
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	rec := httptest.NewRecorder()
	s.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK || rec.Body.String() != "<html>page</html>" {
		t.Fatalf("index: %d %q", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/html") {
		t.Fatalf("content type = %q", ct)
	}
}

func TestServeOnPortZero(t *testing.T) {
	// End-to-end: bind port 0, discover the real port, and call the API.
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := ln.Addr().String()
	ln.Close()

	s := newTestServer(t)
	ln, err = net.Listen("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	srv := &http.Server{Handler: s}
	go func() { _ = srv.Serve(ln) }()
	t.Cleanup(func() { _ = srv.Close() })

	resp, err := http.Post("http://"+ln.Addr().String()+"/api/run",
		"application/json", strings.NewReader(goodBody))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}
}
