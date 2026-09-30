package main

import (
	"embed"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
)

//go:embed index.html
var content embed.FS

func main() {
	host := flag.String("host", "127.0.0.1", "监听地址（如 0.0.0.0 或 127.0.0.1）")
	port := flag.Int("port", 8080, "监听端口；0 表示由系统分配")
	flag.Parse()

	mux := http.NewServeMux()
	mux.HandleFunc("GET /", indexHandler)
	mux.HandleFunc("POST /api/compute", computeHandler)

	addr := fmt.Sprintf("%s:%d", *host, *port)
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "无法监听 %s: %v\n", addr, err)
		os.Exit(1)
	}
	actual := ln.Addr().(*net.TCPAddr)
	fmt.Printf("数值实验台已启动: http://%s:%d\n", *host, actual.Port)
	log.Printf("监听 %s（实际端口 %d）", addr, actual.Port)

	srv := &http.Server{Addr: addr, Handler: mux}
	if err := srv.Serve(ln); err != nil && err != http.ErrServerClosed {
		log.Fatal(err)
	}
}

func indexHandler(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	data, err := content.ReadFile("index.html")
	if err != nil {
		http.Error(w, "页面缺失", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write(data)
}

func computeHandler(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 4<<20))
	if err != nil {
		writeError(w, http.StatusBadRequest, "无法读取请求体: "+err.Error())
		return
	}
	var req ExperimentRequest
	if err := json.Unmarshal(body, &req); err != nil {
		writeError(w, http.StatusBadRequest, "请求 JSON 无法解析: "+err.Error())
		return
	}
	resp, err := Compute(&req)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	_ = json.NewEncoder(w).Encode(resp)
}

type errBody struct {
	Error string `json:"error"`
}

func writeError(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(errBody{Error: msg})
}
