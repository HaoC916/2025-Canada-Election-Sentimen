import type { CorrelationRow, Party } from '../types'

const PARTIES: Party[] = ['Liberal', 'Conservative', 'NDP']

function cellColor(r: number | null) {
  if (r === null) return { background: '#f1f1ef', color: '#9a9a95' }
  const intensity = Math.min(Math.abs(r), 1)
  const bg =
    r >= 0
      ? `rgba(44, 123, 182, ${0.12 + intensity * 0.55})`
      : `rgba(215, 25, 28, ${0.12 + intensity * 0.55})`
  const color = intensity > 0.55 ? '#fff' : '#1c1c1e'
  return { background: bg, color }
}

interface Props {
  rows: CorrelationRow[]
  lags: number[]
  model: 'vader' | 'trans'
  title: string
  lagUnit: string
}

export default function CorrelationGrid({ rows, lags, model, title, lagUnit }: Props) {
  const field = model === 'vader' ? 'vaderPearson' : 'transPearson'

  return (
    <div className="overflow-x-auto">
      <p className="mb-2 text-sm font-medium text-neutral-600">{title}</p>
      <table className="w-full min-w-[420px] border-collapse text-sm">
        <thead>
          <tr>
            <th className="p-2 text-left font-medium text-neutral-500">Party</th>
            {lags.map((lag) => (
              <th key={lag} className="p-2 text-center font-medium text-neutral-500">
                {lag === 0 ? 'Same ' + lagUnit : `+${lag} ${lagUnit}${lag > 1 ? 's' : ''}`}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {PARTIES.map((party) => (
            <tr key={party}>
              <td className="p-2 font-medium">{party}</td>
              {lags.map((lag) => {
                const row = rows.find((r) => r.party === party && r.lag === lag)
                const value = row ? row[field] : null
                const style = cellColor(value)
                return (
                  <td key={lag} className="p-1 text-center">
                    <div
                      className="rounded-md py-1.5 font-mono text-xs tabular-nums"
                      style={style}
                    >
                      {value === null ? '—' : value.toFixed(2)}
                    </div>
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
