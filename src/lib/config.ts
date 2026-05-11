export const siteUrl = process.env.NEXT_PUBLIC_SITE_URL ?? "https://inspired-minds.vercel.app";

export const config = {
  site: {
    title: "My Inspired-Minds",
    name: "My Inspired-Minds",
    description: "Record inspired moments and thoughts",
    keywords: ["Deep Learning", "AI", "Computer Vision"],
    url: siteUrl,
    baseUrl: siteUrl,
    image: `${siteUrl}/og-image.png`,
    favicon: {
      ico: "/favicon.ico",
      png: "/favicon.png",
      svg: "/favicon.svg",
      appleTouchIcon: "/favicon.png",
    },
    manifest: "/site.webmanifest",
    rss: {
      title: "My Inspired-Minds",
      description: "Record inspired moments and thoughts",
      feedLinks: {
        rss2: "/rss.xml",
        json: "/feed.json",
        atom: "/atom.xml",
      },
    },
  },
  author: {
    name: "FisherCat",
    email: "fishercat_@outlook.com",
    bio: "A Graduated student pursuing a Master in computer vision and deep learning, with a passion for expolring and coding.",
  },
  social: {
    github: "https://github.com/FisherCats",
    // x: "https://x.com/xxx",
    // xiaohongshu: "https://www.xiaohongshu.com/user/profile/xxx",
    // wechat: "https://example.com/images/wechat-official-account.png",
    // buyMeACoffee: "https://www.buymeacoffee.com/xxx",
  },
  giscus: {
    repo: "guangzhengli/hugo-ladder-exampleSite",
    repoId: "R_kgDOHyVOjg",
    categoryId: "DIC_kwDOHyVOjs4CQsH7",
  },
  navigation: {
    main: [
      { 
        title: "Notes", 
        href: "/blog",
      },
      {
        title: "Paper",
        href: "/papers",
      }
    ],
  },
  seo: {
    metadataBase: new URL(siteUrl),
    alternates: {
      canonical: './',
    },
    openGraph: {
      type: "website" as const,
      locale: "zh_CN",
    },
    twitter: {
      card: "summary_large_image" as const,
      creator: "@FisherCats",
    },
  },
};
