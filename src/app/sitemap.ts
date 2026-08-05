import type { MetadataRoute } from 'next'
import { allBlogs, allPapers, allTutorials } from 'content-collections'
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
    {
      url: `${config.site.url}/tutorials`,
      lastModified: new Date(),
      changeFrequency: 'weekly',
      priority: 0.8,
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

  const tutorials = allTutorials.map((tutorial: any) => ({
    url: `${config.site.url}/tutorials/${tutorial.slug}`,
    lastModified: new Date(tutorial.updated ?? tutorial.date),
    changeFrequency: 'weekly' as const,
    priority: 0.7,
  }))

  return [...pages, ...blogs, ...papers, ...tutorials]
}
