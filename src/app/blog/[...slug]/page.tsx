import { allBlogs } from "content-collections"
import type { Metadata } from "next"
import Link from "next/link"
import { absoluteUrl } from "@/lib/utils"
import { notFound } from "next/navigation"
import { getTableOfContents } from "@/lib/toc"
import { DashboardTableOfContents } from "@/components/toc"
import { MDXRemote } from 'next-mdx-remote-client/rsc'
import count from 'word-count'
import { components } from "@/components/mdx-components"
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import rehypeHighlight from 'rehype-highlight';
import rehypeSlug from 'rehype-slug';
import 'highlight.js/styles/github-dark.min.css'
// import GiscusComments from "@/components/giscus-comments"
import { GoToTop } from "@/components/go-to-top"
import 'katex/dist/katex.min.css';
import { config } from "@/lib/config";

type BlogsPageProps = {
  params: Promise<{slug: string[]}>
  searchParams: Promise<{ [key: string]: string | string[] | undefined }>
}

const options = {
  mdxOptions: {
      remarkPlugins: [remarkGfm, remarkMath],
      rehypePlugins: [
        rehypeKatex,
        rehypeHighlight,
        rehypeSlug
      ],
  }
}

async function getBlogsFromParams(slugs: string[]) {
  const slug = slugs?.join("/") || ""
  const blog = allBlogs.find((blog: any) => blog.slug === slug)

  if (!blog) {
    return null
  }

  return blog
}

export async function generateMetadata({ params }: BlogsPageProps): Promise<Metadata> {
  const { slug } = await params
  const blog = await getBlogsFromParams(slug)

  if (!blog) {
    return {}
  }

  return {
    title: blog.title,
    description: blog.title,
    keywords: blog.keywords,
    openGraph: {
      title: blog.title,
      description: blog.title,
      type: config.seo.openGraph.type,
      url: absoluteUrl("/" + blog.slug),
      images: [
        {
          url: config.site.image
        },
      ],
    },
    twitter: {
      card: config.seo.twitter.card,
      title: blog.title,
      description: blog.title,
      images: [
        {
          url: config.site.image
        },
      ],
      creator: config.seo.twitter.creator,
    },
  }
}

export async function generateStaticParams(): Promise<string[]> {
  // @ts-ignore
  return allBlogs.map((blog: any) => ({
    slug: blog.slug.split('/'),
  }))
}

export default async function BlogPage(props: BlogsPageProps) {
  const { slug } = await props.params;
  const blog = await getBlogsFromParams(slug)

  if (!blog) {
    notFound()
  }

  const toc = await getTableOfContents(blog.content)
  const sortedBlogs = [...allBlogs].sort((a: any, b: any) => new Date(b.date).getTime() - new Date(a.date).getTime())
  const currentIndex = sortedBlogs.findIndex((item: any) => item.slug === blog.slug)
  const newerBlog = currentIndex > 0 ? sortedBlogs[currentIndex - 1] : null
  const olderBlog = currentIndex >= 0 && currentIndex < sortedBlogs.length - 1 ? sortedBlogs[currentIndex + 1] : null

  return (
    <main className="relative mx-auto w-full max-w-7xl px-4 py-6 sm:px-6 lg:gap-12 lg:py-8 xl:grid xl:grid-cols-[minmax(0,1fr)_240px]">
      <article className="w-full min-w-0">
        <div className="my-8">
          <h1 className="text-[2.2rem] font-bold leading-tight sm:text-[2.7rem]">{blog.title}</h1>
        </div>

        <div className="my-4">
          <p className="text-sm">
            {(() => {
              const d = new Date(blog.date);
              return `${d.getFullYear()}年${d.getMonth() + 1}月${d.getDate()}日`;
            })()} · {count(blog.content)} 字
          </p>
        </div>

        <div className="min-w-0 overflow-hidden">
          <MDXRemote source={blog.content} components={components} options={options} />
        </div>

        {(olderBlog || newerBlog) ? (
          <nav className="mt-16 grid gap-4 border-t border-border pt-8 sm:grid-cols-2" aria-label="Post navigation">
            {olderBlog ? (
              <Link
                href={`/blog/${olderBlog.slug}`}
                className="rounded-md border border-border p-4 transition-colors hover:bg-accent hover:text-accent-foreground"
              >
                <span className="block text-sm text-muted-foreground">上一篇</span>
                <span className="mt-1 block text-lg font-semibold">{olderBlog.title}</span>
              </Link>
            ) : (
              <div />
            )}
            {newerBlog ? (
              <Link
                href={`/blog/${newerBlog.slug}`}
                className="rounded-md border border-border p-4 text-right transition-colors hover:bg-accent hover:text-accent-foreground"
              >
                <span className="block text-sm text-muted-foreground">下一篇</span>
                <span className="mt-1 block text-lg font-semibold">{newerBlog.title}</span>
              </Link>
            ) : null}
          </nav>
        ) : null}

        {/* <GiscusComments /> */}
      </article>
      <div className="hidden text-base xl:block">
        <div className="sticky top-16 -mt-6 h-[calc(100vh-3.5rem)]">
          <div className="h-full overflow-auto pb-10 flex flex-col justify-between mt-16 [&::-webkit-scrollbar]:hidden [-ms-overflow-style:'none'] [scrollbar-width:'none']">
            <DashboardTableOfContents toc={toc} />
            <GoToTop />
          </div>
        </div>
      </div>
    </main>
  );
}
