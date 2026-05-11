import type { MetadataRoute } from 'next'
import { allBlogs, allPapers } from 'content-collections'
import { config } from '@/lib/config'

export default function sitemap(): MetadataRoute.Sitemap {
  const pages: MetadataRoute.Sitemap = [
    {
      url: config.site.url,
      lastModified: new Date(),
      changeFrequency: 'daily',
      priority: 1,
    },
    {
      url: `${config.site.url}/blog`,
      lastModified: new Date(),
      changeFrequency: 'daily',
      priority: 0.9,
    },
  ]

  const blogs = allBlogs.map((blog: any) => ({
    url: `${config.site.url}/blog/${blog.slug}`,
    lastModified: new Date(blog.updated ?? blog.date),
    changeFrequency: 'weekly' as const,
    priority: 0.8,
  }))

  const papers = allPapers.map((paper: any) => ({
    url: `${config.site.url}/papers/${paper.slug}`,
    lastModified: new Date(paper.updated ?? paper.date),
    changeFrequency: 'weekly' as const,
    priority: 0.7,
  }))

  return [...pages, ...blogs, ...papers]
}
