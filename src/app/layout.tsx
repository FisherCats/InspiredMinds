import type { Metadata } from "next";
import "./globals.css";
import { Header } from "@/components/header";
import { config } from "@/lib/config";

export const metadata: Metadata = {
  title: config.site.title,
  description: config.site.description,
  keywords: config.site.keywords,
  metadataBase: config.seo.metadataBase,
  alternates: config.seo.alternates,
  icons: [
    { rel: "icon", url: config.site.favicon.png, sizes: "any", type: "image/png" },
    { rel: "apple-touch-icon", url: config.site.favicon.appleTouchIcon, sizes: "180x180" },
  ],
  openGraph: {
    url: config.site.url,
    type: config.seo.openGraph.type,
    title: config.site.title,
    description: config.site.description,
    images: [
      { url: config.site.image }
    ]
  },
  twitter: {
    site: config.site.url,
    card: config.seo.twitter.card,
    title: config.site.title,
    description: config.site.description,
    images: [
      { url: config.site.image }
    ]
  },
  manifest: config.site.manifest,
  appleWebApp: {
    title: config.site.title,
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                try {
                  var theme = localStorage.getItem("theme");
                  var prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
                  var isDark = theme ? theme === "dark" : prefersDark;
                  document.documentElement.classList.toggle("dark", isDark);
                  document.documentElement.style.colorScheme = isDark ? "dark" : "light";
                } catch (_) {}
              })();
            `,
          }}
        />
        <style>
          {`
            body {
              font-family: "Book Antiqua", "Palatino Linotype", "SimSun", "宋体", serif;
            }
          `}
        </style>
        <link rel="alternate" type="application/rss+xml" title="RSS" href="/rss.xml" />
        <link rel="alternate" type="application/atom+xml" title="Atom" href="/atom.xml" />
        <link rel="alternate" type="application/json" title="JSON" href="/feed.json" />
        <link rel="icon" href={config.site.favicon.png} sizes="any" type="image/png" />
      </head>
      <body className="min-w-md overflow-x-hidden">
        <Header />
        {children}
      </body>
    </html>
  );
}
