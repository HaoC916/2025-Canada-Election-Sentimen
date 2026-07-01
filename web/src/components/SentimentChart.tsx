import {
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { SeriesPoint } from '../utils/series'
import type { Party } from '../types'
import { PARTY_COLOR } from '../types'

interface Props {
  data: SeriesPoint[]
  party: Party
  model: 'vader' | 'trans'
  electionDay: string
  finalResult: number
  dark: boolean
}

const dayMs = 24 * 60 * 60 * 1000

function fmtDate(ts: number) {
  return new Date(ts).toLocaleDateString('en-CA', { month: 'short', day: 'numeric' })
}

export default function SentimentChart({
  data,
  party,
  model,
  electionDay,
  finalResult,
  dark,
}: Props) {
  const sentKey = model === 'vader' ? 'vaderAvg' : 'transPosRatio'
  const rows = data.map((d) => ({ ...d, ts: new Date(d.date).getTime() }))
  const electionTs = new Date(electionDay).getTime()
  const minTs = rows.length ? rows[0].ts : electionTs
  const maxTs = rows.length ? rows[rows.length - 1].ts : electionTs
  const showElectionMarkers = electionTs >= minTs - dayMs && electionTs <= maxTs + dayMs

  const gridStroke = dark ? '#3a3b3e' : '#e9e8e3'
  const tickFill = dark ? '#9a9b9e' : '#6b6b66'
  const markerStroke = dark ? '#e7e7e5' : '#1c1c1e'
  const sentimentLine = dark ? '#a78bfa' : '#7c3aed'

  return (
    <div>
      {showElectionMarkers && (
        <p className="mb-1 flex items-center gap-1.5 text-xs text-neutral-500 dark:text-neutral-400">
          <span
            className="inline-block h-2 w-2 rounded-full bg-neutral-900 dark:bg-neutral-100"
            aria-hidden="true"
          />
          Election day marker — final result {finalResult}%
        </p>
      )}
      <ResponsiveContainer width="100%" height={360}>
        <ComposedChart data={rows} margin={{ top: 12, right: 40, left: 4, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
          <XAxis
            dataKey="ts"
            type="number"
            domain={['dataMin', 'dataMax']}
            tickFormatter={fmtDate}
            tick={{ fontSize: 12, fill: tickFill }}
          />
          <YAxis
            yAxisId="sentiment"
            tick={{ fontSize: 12, fill: tickFill }}
            width={50}
            label={{
              value: model === 'vader' ? 'VADER score' : 'Transformer pos. ratio',
              angle: -90,
              position: 'insideLeft',
              style: { fontSize: 11, fill: tickFill },
            }}
          />
          <YAxis
            yAxisId="polling"
            orientation="right"
            tick={{ fontSize: 12, fill: tickFill }}
            width={56}
            label={{
              value: 'Polling %',
              angle: 90,
              position: 'insideRight',
              style: { fontSize: 11, fill: tickFill },
            }}
          />
          <Tooltip
            contentStyle={
              dark
                ? { background: '#242527', border: '1px solid #3a3b3e', color: '#e7e7e5' }
                : undefined
            }
            labelFormatter={(ts) => fmtDate(Number(ts))}
            formatter={(value, name) =>
              [typeof value === 'number' ? value.toFixed(3) : String(value), String(name)] as [
                string,
                string,
              ]
            }
          />
          <Legend wrapperStyle={{ fontSize: 12, color: tickFill }} />
          {showElectionMarkers && (
            <ReferenceLine
              yAxisId="sentiment"
              x={electionTs}
              stroke={markerStroke}
              strokeDasharray="4 4"
            />
          )}
          {showElectionMarkers && (
            <ReferenceDot
              yAxisId="polling"
              x={electionTs}
              y={finalResult}
              r={5}
              fill={markerStroke}
              stroke="none"
            />
          )}
          <Line
            yAxisId="sentiment"
            type="monotone"
            dataKey={sentKey}
            name={`${party} sentiment`}
            stroke={sentimentLine}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
          <Line
            yAxisId="polling"
            type="monotone"
            dataKey="polling"
            name={`${party} polling %`}
            stroke={PARTY_COLOR[party]}
            strokeWidth={2}
            dot={false}
            connectNulls
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
