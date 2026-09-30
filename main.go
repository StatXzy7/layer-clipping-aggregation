// Command layerbench serves a local web UI for the layered clipping /
// Gaussian-noise aggregation / layer-wise scaling experiment.
package main

import (
	"context"
	"embed"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"layerbench/server"
)

//go:embed web/index.html
var webFS embed.FS

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := run(os.Args[1:], os.Stdout, ctx.Done()); err != nil {
		log.Fatal(err)
	}
}

// run parses flags, binds the listener and serves until the stop channel
// closes. It prints the actual listening URL (important when port is 0).
func run(args []string, out io.Writer, stop <-chan struct{}) error {
	fs := flag.NewFlagSet("layerbench", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	host := fs.String("host", "127.0.0.1", "listen host/address")
	port := fs.Int("port", 8080, "listen port (0 picks a free port)")
	if err := fs.Parse(args); err != nil {
		return err
	}

	page, err := webFS.ReadFile("web/index.html")
	if err != nil {
		return fmt.Errorf("read embedded page: %w", err)
	}

	addr := net.JoinHostPort(*host, fmt.Sprint(*port))
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf("listen %s: %w", addr, err)
	}

	srv := &http.Server{Handler: server.New(page)}
	errCh := make(chan error, 1)
	go func() { errCh <- srv.Serve(ln) }()

	actualPort := ln.Addr().(*net.TCPAddr).Port
	fmt.Fprintf(out, "layerbench listening on http://%s/\n",
		net.JoinHostPort(*host, fmt.Sprint(actualPort)))

	<-stop
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = srv.Shutdown(ctx)
	if err := <-errCh; err != nil && err != http.ErrServerClosed {
		return fmt.Errorf("serve: %w", err)
	}
	return nil
}
