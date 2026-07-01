import { useEffect, useMemo, useState } from 'react'
import SentimentChart from './components/SentimentChart'
import VolumeChart from './components/VolumeChart'
import CorrelationGrid from './components/CorrelationGrid'
import StatCards from './components/StatCards'
import ThemeToggle from './components/ThemeToggle'
import { buildDailySeries, buildWeeklySeries, fmtCompact } from './utils/series'
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
const THEME_KEY = 'election-sentiment-theme'

function segButton(active: boolean) {
  return `px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${
    active
      ? 'bg-neutral-900 text-white dark:bg-neutral-100 dark:text-neutral-900'
      : 'bg-white text-neutral-600 hover:bg-neutral-100 dark:bg-neutral-800 dark:text-neutral-300 dark:hover:bg-neutral-700'
  }`
}

function getInitialTheme(): boolean {
  if (typeof window === 'undefined') return false
  const stored = window.localStorage.getItem(THEME_KEY)
  return stored === 'dark'
}

export default function App() {
  const [party, setParty] = useState<Party>('Liberal')
  const [model, setModel] = useState<'vader' | 'trans'>('trans')
  const [timescale, setTimescale] = useState<'weekly' | 'daily'>('weekly')
  const [dark, setDark] = useState<boolean>(getInitialTheme)

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    window.localStorage.setItem(THEME_KEY, dark ? 'dark' : 'light')
  }, [dark])

  const series = useMemo(() => {
    return timescale === 'weekly'
      ? buildWeeklySeries(weekly, pollWeekly, party)
      : buildDailySeries(daily, pollDaily, party)
  }, [timescale, party])

  const finalResult = meta.finalResults[POLL_COL[party]]

  return (
    <div className="min-h-screen bg-[#f7f7f5] dark:bg-[#17181a]">
      <div className="mx-auto max-w-4xl px-5 py-10">
        <header className="mb-8">
          <div className="flex items-start justify-between gap-4">
            <h1 className="text-3xl font-semibold tracking-tight text-neutral-900 sm:text-4xl dark:text-neutral-50">
              Canada Election Sentiment Explorer 🇨🇦
            </h1>
            <ThemeToggle dark={dark} onToggle={() => setDark((d) => !d)} />
          </div>
          <p className="mt-1 max-w-2xl text-[15px] text-neutral-600 dark:text-neutral-400">
            Does a party&apos;s mood on Reddit actually track how it polls?
          </p>

          <div className="mt-5">
            <StatCards meta={meta} />
          </div>

          <div className="mt-3 rounded-xl border border-neutral-200 bg-neutral-50 px-4 py-3 text-sm text-neutral-700 dark:border-neutral-700 dark:bg-neutral-800 dark:text-neutral-300">
            <span className="font-semibold text-neutral-900 dark:text-neutral-50">
              The finding:{' '}
            </span>
            weekly sentiment tracked the polls for Liberals (r&nbsp;=&nbsp;0.81) and
            Conservatives (r&nbsp;=&nbsp;−0.68) — but day-to-day noise washed the signal out
            entirely.
          </div>
        </header>

        <section className="mb-6 rounded-2xl border border-neutral-200 bg-white p-4 sm:p-5 dark:border-neutral-700 dark:bg-neutral-900">
          <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
            <div className="flex items-center gap-1.5 rounded-full bg-neutral-100 p-1 dark:bg-neutral-800">
              {PARTIES.map((p) => (
                <button key={p} className={segButton(party === p)} onClick={() => setParty(p)}>
                  {p}
                </button>
              ))}
            </div>
            <div className="flex items-center gap-1.5 rounded-full bg-neutral-100 p-1 dark:bg-neutral-800">
              {(['trans', 'vader'] as const).map((m) => (
                <button key={m} className={segButton(model === m)} onClick={() => setModel(m)}>
                  {m === 'trans' ? 'Transformer' : 'VADER'}
                </button>
              ))}
            </div>
            <div className="flex items-center gap-1.5 rounded-full bg-neutral-100 p-1 dark:bg-neutral-800">
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
              dark={dark}
            />
          </div>

          <div className="mt-2">
            <p className="mb-1 text-xs font-medium uppercase tracking-wide text-neutral-400 dark:text-neutral-500">
              Comment volume
            </p>
            <VolumeChart data={series} party={party} dark={dark} />
          </div>
        </section>

        <section className="mb-6 rounded-2xl border border-neutral-200 bg-white p-4 sm:p-5 dark:border-neutral-700 dark:bg-neutral-900">
          <h2 className="mb-1 text-lg font-semibold text-neutral-900 dark:text-neutral-50">
            Does Reddit sentiment track the polls?
          </h2>
          <p className="mb-4 text-sm text-neutral-600 dark:text-neutral-400">
            Pearson correlation between each party&apos;s sentiment series and its polling
            average, at increasing lag. Blue = sentiment moves with the poll, red = it moves
            against it. Weekly aggregation cancels out day-to-day noise; daily doesn&apos;t.
          </p>
          <div className="grid gap-6">
            <CorrelationGrid
              rows={correlations.weekly}
              lags={[0, 1]}
              model={model}
              lagUnit="week"
              title={`Weekly · ${model === 'vader' ? 'VADER' : 'Transformer'}`}
              dark={dark}
            />
            <CorrelationGrid
              rows={correlations.daily}
              lags={[0, 1, 2, 3, 4, 5]}
              model={model}
              lagUnit="day"
              title={`Daily · ${model === 'vader' ? 'VADER' : 'Transformer'}`}
              dark={dark}
            />
          </div>
        </section>

        <footer className="pb-6 text-xs leading-relaxed text-neutral-500 dark:text-neutral-500">
          <p>
            Real pipeline output, not reconstructed — {fmtCompact(meta.commentsScanned)} comments
            scanned, {fmtCompact(meta.targetedSentences)} party-targeted sentences scored with
            VADER + CardiffNLP RoBERTa, aggregated into {meta.weeklyWeekRange[1]} weekly and{' '}
            {meta.dailyBucketCount} daily buckets and correlated against public polling.
          </p>
          <p className="mt-3">
            Built by Ryan Chen, with teammates Luna Sang and Zili Ding (CMPT 732 group project,
            SFU).
          </p>
          <p className="mt-1">
            Source:{' '}
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
