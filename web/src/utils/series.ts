import type {
  DailySentiment,
  Party,
  PollDaily,
  PollWeekly,
  WeeklySentiment,
} from '../types'
import { PARTY_POLL_COL } from '../types'

export interface SeriesPoint {
  date: string
  label: string
  vaderAvg: number
  transPosRatio: number
  polling: number | null
  volume: number
}

export function fmtCompact(n: number) {
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(2)}B`
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`
  return n.toLocaleString()
}

const WEEK_1_START = new Date('2025-01-06T00:00:00Z')

function weekToDate(week: number): string {
  const d = new Date(WEEK_1_START)
  d.setUTCDate(d.getUTCDate() + (week - 1) * 7)
  return d.toISOString().slice(0, 10)
}

export function buildWeeklySeries(
  weekly: WeeklySentiment[],
  pollWeekly: PollWeekly[],
  party: Party,
): SeriesPoint[] {
  const pollCol = PARTY_POLL_COL[party]
  const pollByWeek = new Map(pollWeekly.map((p) => [p.week, p]))

  return weekly
    .filter((r) => r.party === party)
    .sort((a, b) => a.week - b.week)
    .map((r) => {
      const poll = pollByWeek.get(r.week)
      const date = weekToDate(r.week)
      return {
        date,
        label: `Week ${r.week}`,
        vaderAvg: r.vaderAvg,
        transPosRatio: r.transPosRatio,
        polling: poll ? poll[pollCol] : null,
        volume: r.commentVolume,
      }
    })
}

export function buildDailySeries(
  daily: DailySentiment[],
  pollDaily: PollDaily[],
  party: Party,
): SeriesPoint[] {
  const pollCol = PARTY_POLL_COL[party]
  const pollByDate = new Map(pollDaily.map((p) => [p.date, p]))

  return daily
    .filter((r) => r.party === party)
    .sort((a, b) => a.date.localeCompare(b.date))
    .map((r) => {
      const poll = pollByDate.get(r.date)
      return {
        date: r.date,
        label: r.date,
        vaderAvg: r.vaderAvg,
        transPosRatio: r.transPosRatio,
        polling: poll ? poll[pollCol] : null,
        volume: r.commentVolume,
      }
    })
}
