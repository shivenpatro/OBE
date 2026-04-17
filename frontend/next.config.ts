import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        // Proxy all /api/* calls from the Next.js dev server to the
        // FastAPI backend running on port 8000.  This eliminates CORS
        // entirely — the browser only ever talks to localhost:3000.
        source: "/api/:path*",
        destination: "http://localhost:8000/api/:path*",
      },
    ];
  },
};

export default nextConfig;
