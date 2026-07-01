import type { CorrelationRow, Party } from '../types'

const PARTIES: Party[] = ['Liberal', 'Conservative', 'NDP']

function cellColor(r: number | null, dark: boolean) {
  if (r === null) {
    return dark
      ? { background: '#2a2b2e', color: '#8a8b8e' }
      : { background: '#f1f1ef', color: '#9a9a95' }
  }
  const intensity = Math.min(Math.abs(r), 1)
  const alpha = dark ? 0.18 + intensity * 0.55 : 0.12 + intensity * 0.55
  const bg =
    r >= 0 ? `rgba(74, 149, 209, ${alpha})` : `rgba(224, 76, 78, ${alpha})`
  const color = dark ? '#fff' : intensity > 0.55 ? '#fff' : '#1c1c1e'
  return { background: bg, color }
}

interface Props {
  rows: CorrelationRow[]
  lags: number[]
  model: 'vader' | 'trans'
  title: string
  lagUnit: string
  dark: boolean
}

export default function CorrelationGrid({ rows, lags, model, title, lagUnit, dark }: Props) {
  const field = model === 'vader' ? 'vaderPearson' : 'transPearson'

  return (
    <div className="overflow-x-auto">
      <p className="mb-1.5 text-sm font-medium text-neutral-600 dark:text-neutral-400">{title}</p>
      <table className="border-collapse text-sm">
        <thead>
          <tr>
            <th className="w-20 p-2 pl-0 text-left font-medium text-neutral-500 sm:w-28 dark:text-neutral-400">
              Party
            </th>
            {lags.map((lag) => (
              <th
                key={lag}
                className="w-16 p-2 text-center font-medium text-neutral-500 sm:w-20 dark:text-neutral-400"
              >
                {lag === 0 ? 'Same ' + lagUnit : `+${lag} ${lagUnit}${lag > 1 ? 's' : ''}`}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {PARTIES.map((party) => (
            <tr key={party}>
              <td className="p-2 pl-0 font-medium text-neutral-800 dark:text-neutral-200">
                {party}
              </td>
              {lags.map((lag) => {
                const row = rows.find((r) => r.party === party && r.lag === lag)
                const value = row ? row[field] : null
                const style = cellColor(value, dark)
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
