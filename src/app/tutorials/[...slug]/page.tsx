import type { Metadata } from "next"
import Link from "next/link"
import { notFound } from "next/navigation"
import { allTutorials } from "content-collections"
import { MDXRemote } from "next-mdx-remote-client/rsc"
import count from "word-count"
import remarkGfm from "remark-gfm"
import remarkMath from "remark-math"
import rehypeHighlight from "rehype-highlight"
import rehypeKatex from "rehype-katex"
import rehypeSlug from "rehype-slug"
import "highlight.js/styles/github-dark.min.css"
import "katex/dist/katex.min.css"

import { components } from "@/components/mdx-components"
import { GoToTop } from "@/components/go-to-top"
import { DashboardTableOfContents } from "@/components/toc"
import { config } from "@/lib/config"
import { getTableOfContents } from "@/lib/toc"
import { absoluteUrl } from "@/lib/utils"

type TutorialPageProps = {
  params: Promise<{ slug: string[] }>
}

const options = {
  mdxOptions: {
    remarkPlugins: [remarkGfm, remarkMath],
    rehypePlugins: [rehypeKatex, rehypeHighlight, rehypeSlug],
  },
}

function getTutorialFromParams(slugs: string[]) {
  const slug = slugs?.join("/") || ""
  return allTutorials.find((tutorial) => tutorial.slug === slug) ?? null
}

function formatDate(date: string) {
  const value = new Date(date)
  return `${value.getFullYear()}年${value.getMonth() + 1}月${value.getDate()}日`
}

export async function generateMetadata({ params }: TutorialPageProps): Promise<Metadata> {
  const { slug } = await params
  const tutorial = getTutorialFromParams(slug)

  if (!tutorial) return {}

  const description = tutorial.summary ?? tutorial.title
  const url = absoluteUrl(`/tutorials/${tutorial.slug}`)

  return {
    title: tutorial.title,
    description,
    keywords: tutorial.keywords,
    openGraph: {
      title: tutorial.title,
      description,
      type: config.seo.openGraph.type,
      url,
      images: [{ url: config.site.image }],
    },
    twitter: {
      card: config.seo.twitter.card,
      title: tutorial.title,
      description,
      images: [{ url: config.site.image }],
      creator: config.seo.twitter.creator,
    },
  }
}

export function generateStaticParams() {
  return allTutorials.map((tutorial) => ({
    slug: tutorial.slug.split("/"),
  }))
}

export default async function TutorialPage({ params }: TutorialPageProps) {
  const { slug } = await params
  const tutorial = getTutorialFromParams(slug)

  if (!tutorial) notFound()

  const toc = await getTableOfContents(tutorial.content)
  const sortedTutorials = [...allTutorials].sort(
    (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()
  )
  const currentIndex = sortedTutorials.findIndex((item) => item.slug === tutorial.slug)
  const newerTutorial = currentIndex > 0 ? sortedTutorials[currentIndex - 1] : null
  const olderTutorial =
    currentIndex >= 0 && currentIndex < sortedTutorials.length - 1
      ? sortedTutorials[currentIndex + 1]
      : null

  return (
    <main className="relative mx-auto w-full max-w-7xl px-4 py-6 sm:px-6 lg:gap-12 lg:py-8 xl:grid xl:grid-cols-[minmax(0,1fr)_240px]">
      <article className="w-full min-w-0">
        <div className="my-8">
          <p className="mb-3 text-sm text-muted-foreground">
            <Link href="/tutorials" className="underline underline-offset-4">
              Tutorials
            </Link>
          </p>
          <h1 className="text-[2.2rem] font-bold leading-tight sm:text-[2.7rem]">
            {tutorial.title}
          </h1>
        </div>

        <p className="my-4 text-sm">
          {formatDate(tutorial.date)} · {count(tutorial.content)} 字
        </p>

        <div className="blog-content min-w-0 overflow-hidden">
          <MDXRemote source={tutorial.content} components={components} options={options} />
        </div>

        {olderTutorial || newerTutorial ? (
          <nav
            className="mt-16 grid gap-4 border-t border-border pt-8 sm:grid-cols-2"
            aria-label="教程导航"
          >
            {olderTutorial ? (
              <Link
                href={`/tutorials/${olderTutorial.slug}`}
                className="rounded-md border border-border p-4 transition-colors hover:bg-accent hover:text-accent-foreground"
              >
                <span className="block text-sm text-muted-foreground">上一篇</span>
                <span className="mt-1 block text-lg font-semibold">{olderTutorial.title}</span>
              </Link>
            ) : (
              <div />
            )}
            {newerTutorial ? (
              <Link
                href={`/tutorials/${newerTutorial.slug}`}
                className="rounded-md border border-border p-4 text-right transition-colors hover:bg-accent hover:text-accent-foreground"
              >
                <span className="block text-sm text-muted-foreground">下一篇</span>
                <span className="mt-1 block text-lg font-semibold">{newerTutorial.title}</span>
              </Link>
            ) : null}
          </nav>
        ) : null}
      </article>

      <div className="hidden text-base xl:block">
        <div className="sticky top-16 -mt-6 h-[calc(100vh-3.5rem)]">
          <div className="mt-16 flex h-full flex-col justify-between overflow-auto pb-10 [&::-webkit-scrollbar]:hidden [-ms-overflow-style:'none'] [scrollbar-width:'none']">
            <DashboardTableOfContents toc={toc} />
            <GoToTop />
          </div>
        </div>
      </div>
    </main>
  )
}
