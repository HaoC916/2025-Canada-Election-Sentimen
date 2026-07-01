import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { SeriesPoint } from '../utils/series'
import { PARTY_COLOR } from '../types'
import type { Party } from '../types'

function fmtDate(ts: number) {
  return new Date(ts).toLocaleDateString('en-CA', { month: 'short', day: 'numeric' })
}

export default function VolumeChart({
  data,
  party,
  dark,
}: {
  data: SeriesPoint[]
  party: Party
  dark: boolean
}) {
  const rows = data.map((d) => ({ ...d, ts: new Date(d.date).getTime() }))
  const tickFill = dark ? '#9a9b9e' : '#8a8a85'

  return (
    <ResponsiveContainer width="100%" height={140}>
      <AreaChart data={rows} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
        <XAxis
          dataKey="ts"
          type="number"
          domain={['dataMin', 'dataMax']}
          tickFormatter={fmtDate}
          tick={{ fontSize: 11, fill: tickFill }}
        />
        <YAxis tick={{ fontSize: 11, fill: tickFill }} width={44} />
        <Tooltip
          contentStyle={
            dark
              ? { background: '#242527', border: '1px solid #3a3b3e', color: '#e7e7e5' }
              : undefined
          }
          labelFormatter={(ts) => fmtDate(Number(ts))}
          formatter={(value) =>
            [
              typeof value === 'number' ? value.toLocaleString() : String(value),
              'Targeted comments',
            ] as [string, string]
          }
        />
        <Area
          type="monotone"
          dataKey="volume"
          stroke={PARTY_COLOR[party]}
          fill={PARTY_COLOR[party]}
          fillOpacity={0.15}
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}
