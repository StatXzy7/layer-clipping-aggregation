package main

import (
	"net/http"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"
)

type syncBuffer struct {
	mu  sync.Mutex
	buf strings.Builder
}

func (s *syncBuffer) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.buf.Write(p)
}

func (s *syncBuffer) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.buf.String()
}

func TestRunPortZeroPrintsActualAddressAndServes(t *testing.T) {
	var out syncBuffer
	stop := make(chan struct{})
	errCh := make(chan error, 1)
	go func() { errCh <- run([]string{"--host", "127.0.0.1", "--port", "0"}, &out, stop) }()

	// Wait for the printed URL with the concrete ephemeral port.
	url := ""
	deadline := time.Now().Add(3 * time.Second)
	re := regexp.MustCompile(`http://127\.0\.0\.1:(\d+)/`)
	for time.Now().Before(deadline) {
		if m := re.FindStringSubmatch(out.String()); m != nil {
			url = m[0]
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if url == "" {
		t.Fatalf("no address printed: %q", out.String())
	}
	if strings.HasSuffix(url, ":0/") {
		t.Fatalf("printed port is still zero: %s", url)
	}

	resp, err := http.Get(url)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET status = %d", resp.StatusCode)
	}
	resp.Body.Close()

	close(stop)
	select {
	case err := <-errCh:
		if err != nil {
			t.Fatalf("run returned %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("run did not shut down")
	}
}

func TestRunBadFlag(t *testing.T) {
	var out syncBuffer
	stop := make(chan struct{})
	if err := run([]string{"--nope"}, &out, stop); err == nil {
		t.Fatal("expected flag error")
	}
}

func TestRunBadPort(t *testing.T) {
	var out syncBuffer
	stop := make(chan struct{})
	err := run([]string{"--host", "127.0.0.1", "--port", "-3"}, &out, stop)
	if err == nil || !strings.Contains(err.Error(), "listen") {
		t.Fatalf("expected listen error, got %v", err)
	}
}
