import { useMemo, useState } from 'react'
import SentimentChart from './components/SentimentChart'
import VolumeChart from './components/VolumeChart'
import CorrelationGrid from './components/CorrelationGrid'
import { buildDailySeries, buildWeeklySeries } from './lib/series'
import type { Party } from './types'

import weeklyData from './data/weekly.json'
import dailyData from './data/daily.json'
import pollWeeklyData from './data/pollWeekly.json'
import pollDailyData from './data/pollDaily.json'
import correlationsData from './data/correlations.json'
import metaData from './data/meta.json'
import type {
  CorrelationRow,
  DailySentiment,
  Meta,
  PollDaily,
  PollWeekly,
  WeeklySentiment,
} from './types'

const weekly = weeklyData as WeeklySentiment[]
const daily = dailyData as DailySentiment[]
const pollWeekly = pollWeeklyData as PollWeekly[]
const pollDaily = pollDailyData as PollDaily[]
const correlations = correlationsData as { weekly: CorrelationRow[]; daily: CorrelationRow[] }
const meta = metaData as Meta

const PARTIES: Party[] = ['Liberal', 'Conservative', 'NDP']
const POLL_COL = { Liberal: 'lpc', Conservative: 'cpc', NDP: 'ndp' } as const

function fmtLongDate(iso: string) {
  return new Date(`${iso}T00:00:00Z`).toLocaleDateString('en-CA', {
    month: 'long',
    day: 'numeric',
    year: 'numeric',
    timeZone: 'UTC',
  })
}

function segButton(active: boolean) {
  return `px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${
    active ? 'bg-neutral-900 text-white' : 'bg-white text-neutral-600 hover:bg-neutral-100'
  }`
}

export default function App() {
  const [party, setParty] = useState<Party>('Liberal')
  const [model, setModel] = useState<'vader' | 'trans'>('trans')
  const [timescale, setTimescale] = useState<'weekly' | 'daily'>('weekly')

  const series = useMemo(() => {
    return timescale === 'weekly'
      ? buildWeeklySeries(weekly, pollWeekly, party)
      : buildDailySeries(daily, pollDaily, party)
  }, [timescale, party])

  const finalResult = meta.finalResults[POLL_COL[party]]

  return (
    <div className="min-h-screen bg-[#f7f7f5]">
      <div className="mx-auto max-w-4xl px-5 py-10">
        <header className="mb-8">
          <p className="text-sm font-medium text-neutral-500">CMPT 732 · Big Data · Fall 2025</p>
          <h1 className="mt-1 text-3xl font-semibold tracking-tight text-neutral-900 sm:text-4xl">
            🇨🇦 Election Sentiment Explorer
          </h1>
          <p className="mt-3 max-w-2xl text-[15px] leading-relaxed text-neutral-600">
            {meta.commentsScanned.toLocaleString()} raw Reddit comments (Dec 2024 – Apr 2025),
            filtered down to {meta.targetedSentences.toLocaleString()} sentences that actually
            target a federal party, scored with VADER and a RoBERTa transformer, and checked
            against real polling averages through the {fmtLongDate(meta.electionDay)} election.
          </p>
        </header>

        <section className="mb-6 rounded-2xl border border-neutral-200 bg-white p-4 sm:p-5">
          <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
            <div className="flex items-center gap-1.5 rounded-full bg-neutral-100 p-1">
              {PARTIES.map((p) => (
                <button key={p} className={segButton(party === p)} onClick={() => setParty(p)}>
                  {p}
                </button>
              ))}
            </div>
            <div className="flex items-center gap-1.5 rounded-full bg-neutral-100 p-1">
              {(['trans', 'vader'] as const).map((m) => (
                <button key={m} className={segButton(model === m)} onClick={() => setModel(m)}>
                  {m === 'trans' ? 'Transformer' : 'VADER'}
                </button>
              ))}
            </div>
            <div className="flex items-center gap-1.5 rounded-full bg-neutral-100 p-1">
              {(['weekly', 'daily'] as const).map((t) => (
                <button
                  key={t}
                  className={segButton(timescale === t)}
                  onClick={() => setTimescale(t)}
                >
                  {t === 'weekly' ? 'Weekly' : 'Daily'}
                </button>
              ))}
            </div>
          </div>

          <div className="mt-5">
            <SentimentChart
              data={series}
              party={party}
              model={model}
              electionDay={meta.electionDay}
              finalResult={finalResult}
            />
          </div>

          <div className="mt-2">
            <p className="mb-1 text-xs font-medium uppercase tracking-wide text-neutral-400">
              Comment volume
            </p>
            <VolumeChart data={series} party={party} />
          </div>
        </section>

        <section className="mb-6 rounded-2xl border border-neutral-200 bg-white p-4 sm:p-5">
          <h2 className="mb-1 text-lg font-semibold text-neutral-900">
            Does Reddit sentiment track the polls?
          </h2>
          <p className="mb-4 text-sm text-neutral-600">
            Pearson correlation between each party&apos;s sentiment series and its polling
            average, at increasing lag. Blue = sentiment moves with the poll, red = it moves
            against it. Weekly aggregation cancels out day-to-day noise; daily doesn&apos;t.
          </p>
          <div className="grid gap-6 sm:grid-cols-2">
            <CorrelationGrid
              rows={correlations.weekly}
              lags={[0, 1]}
              model={model}
              lagUnit="week"
              title={`Weekly · ${model === 'vader' ? 'VADER' : 'Transformer'}`}
            />
            <CorrelationGrid
              rows={correlations.daily}
              lags={[0, 1, 2, 3, 4, 5]}
              model={model}
              lagUnit="day"
              title={`Daily · ${model === 'vader' ? 'VADER' : 'Transformer'}`}
            />
          </div>
        </section>

        <footer className="pb-6 text-xs leading-relaxed text-neutral-500">
          <p>
            All numbers here are real pipeline output, not reconstructed:{' '}
            {meta.commentsScanned.toLocaleString()} raw Reddit comments were scanned across four
            monthly dumps, {meta.commentsKeptAfterFilter.toLocaleString()} survived the initial
            filter, and {meta.targetedSentences.toLocaleString()} party-targeted sentences were
            scored with VADER and CardiffNLP&apos;s{' '}
            <code className="rounded bg-neutral-100 px-1 py-0.5">{meta.transformerModel}</code>.
            Aggregated into {meta.weeklyWeekRange[1]} weekly and {meta.dailyBucketCount} daily
            buckets ({meta.dailyDateRange[0]} to {meta.dailyDateRange[1]}), correlated against
            public polling averages.
          </p>
          <p className="mt-2">
            Built by Luna Sang, Ryan Chen, and Zili Ding for CMPT 732 (SFU). Source:{' '}
            <a
              className="underline"
              href="https://github.com/HaoC916/2025-Canada-Election-Sentimen"
              target="_blank"
              rel="noreferrer"
            >
              github.com/HaoC916/2025-Canada-Election-Sentimen
            </a>
            .
          </p>
        </footer>
      </div>
    </div>
  )
}
