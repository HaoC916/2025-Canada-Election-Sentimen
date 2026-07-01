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
}

const dayMs = 24 * 60 * 60 * 1000

function fmtDate(ts: number) {
  return new Date(ts).toLocaleDateString('en-CA', { month: 'short', day: 'numeric' })
}

export default function SentimentChart({ data, party, model, electionDay, finalResult }: Props) {
  const sentKey = model === 'vader' ? 'vaderAvg' : 'transPosRatio'
  const rows = data.map((d) => ({ ...d, ts: new Date(d.date).getTime() }))
  const electionTs = new Date(electionDay).getTime()
  const minTs = rows.length ? rows[0].ts : electionTs
  const maxTs = rows.length ? rows[rows.length - 1].ts : electionTs
  const showElectionMarkers = electionTs >= minTs - dayMs && electionTs <= maxTs + dayMs

  return (
    <ResponsiveContainer width="100%" height={340}>
      <ComposedChart data={rows} margin={{ top: 10, right: 16, left: 0, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e9e8e3" />
        <XAxis
          dataKey="ts"
          type="number"
          domain={['dataMin', 'dataMax']}
          tickFormatter={fmtDate}
          tick={{ fontSize: 12, fill: '#6b6b66' }}
        />
        <YAxis
          yAxisId="sentiment"
          tick={{ fontSize: 12, fill: '#6b6b66' }}
          width={46}
          label={{
            value: model === 'vader' ? 'VADER score' : 'Transformer pos. ratio',
            angle: -90,
            position: 'insideLeft',
            style: { fontSize: 11, fill: '#6b6b66' },
          }}
        />
        <YAxis
          yAxisId="polling"
          orientation="right"
          tick={{ fontSize: 12, fill: '#6b6b66' }}
          width={40}
          label={{
            value: 'Polling %',
            angle: 90,
            position: 'insideRight',
            style: { fontSize: 11, fill: '#6b6b66' },
          }}
        />
        <Tooltip
          labelFormatter={(ts) => fmtDate(Number(ts))}
          formatter={(value, name) =>
            [typeof value === 'number' ? value.toFixed(3) : String(value), String(name)] as [
              string,
              string,
            ]
          }
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {showElectionMarkers && (
          <ReferenceLine
            yAxisId="sentiment"
            x={electionTs}
            stroke="#1c1c1e"
            strokeDasharray="4 4"
            label={{ value: 'Election Day', position: 'top', fontSize: 11 }}
          />
        )}
        {showElectionMarkers && (
          <ReferenceDot
            yAxisId="polling"
            x={electionTs}
            y={finalResult}
            r={5}
            fill="#1c1c1e"
            stroke="none"
            label={{ value: `Final: ${finalResult}%`, position: 'top', fontSize: 11 }}
          />
        )}
        <Line
          yAxisId="sentiment"
          type="monotone"
          dataKey={sentKey}
          name={`${party} sentiment`}
          stroke="#7c3aed"
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
  )
}
