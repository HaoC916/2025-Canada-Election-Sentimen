export type Party = 'Liberal' | 'Conservative' | 'NDP'

export interface WeeklySentiment {
  party: Party
  week: number
  vaderAvg: number
  transPosRatio: number
  transNegRatio: number
  transNeuRatio: number
  commentVolume: number
}

export interface DailySentiment {
  party: Party
  date: string
  vaderAvg: number
  transPosRatio: number
  transNegRatio: number
  transNeuRatio: number
  commentVolume: number
}

export interface PollWeekly {
  week: number
  lpc: number
  cpc: number
  ndp: number
  oth: number
}

export interface PollDaily {
  date: string
  lpc: number
  cpc: number
  ndp: number
  oth: number
  freq: string
  week: number
}

export interface CorrelationRow {
  party: Party
  lag: number
  vaderPearson: number | null
  vaderSpearman: number | null
  vaderN: number
  transPearson: number | null
  transSpearman: number | null
  transN: number
}

export interface Meta {
  commentsScanned: number
  commentsKeptAfterFilter: number
  submissionsScanned: number
  submissionsKeptAfterFilter: number
  targetedSentences: number
  dailyDateRange: [string, string]
  dailyBucketCount: number
  weeklyWeekRange: [number, number]
  electionDay: string
  finalResults: { lpc: number; cpc: number; ndp: number }
  transformerModel: string
}

export const PARTY_POLL_COL: Record<Party, 'lpc' | 'cpc' | 'ndp'> = {
  Liberal: 'lpc',
  Conservative: 'cpc',
  NDP: 'ndp',
}

export const PARTY_COLOR: Record<Party, string> = {
  Liberal: '#d7191c',
  Conservative: '#2c7bb6',
  NDP: '#e8a13a',
}
