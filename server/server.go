// Package server exposes the experiment over a small local HTTP API and
// serves a single editable page. It only uses the Go standard library.
package server

import (
	"encoding/json"
	"log"
	"net/http"
	"sync"

	"layerbench/core"
)

// Server keeps the last successful report so invalid submissions leave the
// displayed result untouched.
type Server struct {
	mu   sync.Mutex
	last *core.Report

	mux *http.ServeMux
}

// New wires the HTTP handlers. indexHTML is the embedded page content.
func New(indexHTML []byte) *Server {
	s := &Server{mux: http.NewServeMux()}
	page := indexHTML
	s.mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write(page)
	})
	s.mux.HandleFunc("GET /api/last", s.handleLast)
	s.mux.HandleFunc("POST /api/run", s.handleRun)
	return s
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.mux.ServeHTTP(w, r)
}

func (s *Server) handleRun(w http.ResponseWriter, r *http.Request) {
	var cfg core.Config
	dec := json.NewDecoder(r.Body)
	if err := dec.Decode(&cfg); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}
	rep, err := core.Run(cfg)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	s.mu.Lock()
	s.last = rep
	s.mu.Unlock()
	writeJSON(w, http.StatusOK, rep)
}

func (s *Server) handleLast(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	rep := s.last
	s.mu.Unlock()
	if rep == nil {
		writeJSON(w, http.StatusOK, map[string]any{"report": nil})
		return
	}
	writeJSON(w, http.StatusOK, rep)
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("write response: %v", err)
	}
}

func writeError(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, map[string]string{"error": msg})
}

// Last returns the most recent successful report (for tests).
func (s *Server) Last() *core.Report {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.last
}
