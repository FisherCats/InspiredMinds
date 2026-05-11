import { type Metadata } from "next";
import { allBlogs } from "content-collections";
import Link from "next/link";
import count from 'word-count'
import { config } from "@/lib/config";

export const metadata: Metadata = {
  title: `Blogs | ${config.site.title}`,
  description: `Blogs of ${config.site.title}`,
  keywords: `${config.site.title}, blogs, ${config.site.title} blogs, nextjs blog template`,
};

type BlogPageProps = {
  searchParams: Promise<{ tag?: string }>
}

export default async function BlogPage({ searchParams }: BlogPageProps) {
  const { tag } = await searchParams
  const selectedTag = tag
  const blogs = [...allBlogs].sort((a: any, b: any) => new Date(b.date).getTime() - new Date(a.date).getTime());
  const tags = Array.from(new Set<string>(blogs.flatMap((blog: any) => blog.tags ?? []))).sort()
  const filteredBlogs = selectedTag
    ? blogs.filter((blog: any) => blog.tags?.includes(selectedTag))
    : blogs

  return (
    <div className="mx-auto w-full max-w-4xl px-4 py-8 sm:px-6">
      <div className="mb-10 flex flex-wrap gap-2">
        <Link
          href="/blog"
          className={`rounded-md border px-3 py-1 text-sm transition-colors ${
            selectedTag
              ? "border-border text-muted-foreground hover:bg-accent hover:text-accent-foreground"
              : "border-foreground bg-foreground text-background"
          }`}
        >
          All
        </Link>
        {tags.map((tag) => (
          <Link
            key={tag}
            href={`/blog?tag=${encodeURIComponent(tag)}`}
            className={`rounded-md border px-3 py-1 text-sm transition-colors ${
              selectedTag === tag
                ? "border-foreground bg-foreground text-background"
                : "border-border text-muted-foreground hover:bg-accent hover:text-accent-foreground"
            }`}
          >
            {tag}
          </Link>
        ))}
      </div>

      <div className="space-y-8">
        {filteredBlogs.map((blog: any) => (
          <article 
            key={blog.slug} 
            className=""
          >
            <Link href={`/blog/${blog.slug}`}>
              <div className="flex flex-col space-y-2">
                <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
                  <h2 className="text-xl font-semibold underline underline-offset-4">
                    {blog.title}
                  </h2>
                    <span className="text-sm text-muted-foreground">
                  {(() => {
                    const d = new Date(blog.date);
                    return `${d.getFullYear()}年${d.getMonth() + 1}月${d.getDate()}日`;
                  })()} · {count(blog.content)} 字
                  </span>
                </div>
                <p className="text-muted-foreground line-clamp-2">
                  {blog.summary}
                </p>
                {blog.tags?.length ? (
                  <div className="flex flex-wrap gap-2 pt-1">
                    {blog.tags.map((tag: string) => (
                      <span key={tag} className="text-xs text-muted-foreground">
                        #{tag}
                      </span>
                    ))}
                  </div>
                ) : null}
              </div>
            </Link>
          </article>
        ))}
      </div>
    </div>
  );
}
