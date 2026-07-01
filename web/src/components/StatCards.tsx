import { FiCalendar, FiCpu, FiFilter, FiTarget } from 'react-icons/fi'
import type { Meta } from '../types'

export function fmtCompact(n: number) {
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(2)}B`
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`
  return n.toLocaleString()
}

export default function StatCards({ meta }: { meta: Meta }) {
  const stats = [
    {
      icon: FiFilter,
      value: fmtCompact(meta.commentsScanned),
      label: 'raw comments scanned',
    },
    {
      icon: FiTarget,
      value: fmtCompact(meta.targetedSentences),
      label: 'party-targeted sentences',
    },
    {
      icon: FiCpu,
      value: 'VADER + RoBERTa',
      label: 'sentiment models, cross-checked',
    },
    {
      icon: FiCalendar,
      value: `${meta.weeklyWeekRange[1]} weeks`,
      label: 'tracked to election day',
    },
  ]

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {stats.map(({ icon: Icon, value, label }) => (
        <div
          key={label}
          className="rounded-xl border border-neutral-200 bg-white p-3 text-center sm:p-4"
        >
          <Icon className="mx-auto mb-1.5 text-neutral-400" size={18} />
          <p className="text-base font-semibold text-neutral-900 sm:text-lg">{value}</p>
          <p className="text-[11px] leading-tight text-neutral-500">{label}</p>
        </div>
      ))}
    </div>
  )
}
