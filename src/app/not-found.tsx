import Link from 'next/link'
import NotFoundIcon from '@/components/icons/not-found'

export default function NotFound() {
  return (
    <main className="flex flex-col items-center justify-center min-h-screen px-4 py-12">
      <div className="text-center mb-16">
        <NotFoundIcon className="mx-auto" />
        <h1 className="text-4xl font-bold mb-4">404 Not Found</h1>
        <p className="text-lg mb-8 text-muted-foreground">
          The page you&apos;re looking for doesn&apos;t exist or has been moved.
        </p>
        <Link href="/" className="inline-block">
          <button className="px-6 py-4 bg-primary text-primary-foreground text-lg rounded-md font-semibold hover:bg-primary/90 hover:shadow-lg transition-all cursor-pointer">
            Return Home
          </button>
        </Link>
      </div>
    </main>
  )
}
