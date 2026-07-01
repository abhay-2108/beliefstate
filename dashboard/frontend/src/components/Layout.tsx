import React, { useState, useCallback, useEffect } from 'react';
import { Select } from './Select';
import { GlobalSearch } from './GlobalSearch';
import { NotificationPanel } from './NotificationPanel';
import { KeyboardShortcuts } from './KeyboardShortcuts';
import { Notification, AlertRule } from '../hooks/useNotifications';

interface Props {
  children: React.ReactNode;
  activeTab: string;
  setActiveTab: (tab: string) => void;
  tabs: { id: string; label: string; icon: string }[];
  sessions: string[];
  selectedSession: string | null;
  setSelectedSession: (session: string | null) => void;
  loading: boolean;
  refreshData: () => void;
  sseConnected: boolean;
  trackingStatus: string;
  notif: {
    notifications: Notification[];
    unreadCount: number;
    markRead: (id: string) => void;
    markAllRead: () => void;
    clearNotifs: () => void;
  };
}

const PAGE_META: Record<string, { title: string; subtitle: string }> = {
  overview:    { title: 'Overview',         subtitle: 'Monitor geopolitical beliefs, timelines, conflicts, and simulation outcomes.' },
  beliefs:     { title: 'Beliefs',          subtitle: 'Browse, search, and manage extracted belief entities, attributes, and values.' },
  timeline:    { title: 'Timeline',         subtitle: 'Track how individual beliefs evolve across conversation turns.' },
  conflicts:   { title: 'Conflicts',        subtitle: 'Review detected contradictions and resolution decisions.' },
  activity:    { title: 'Activity',         subtitle: 'Real-time event stream of tracking, extraction, and simulation activity.' },
  compare:     { title: 'Compare Sessions', subtitle: 'Diff two sessions side-by-side to identify changes and gaps.' },
  simulator:   { title: 'Simulator',        subtitle: 'Test extraction, context injection, and timing with custom inputs.' },
  settings:    { title: 'Settings',         subtitle: 'Configure thresholds, strategies, providers, and extraction prompts.' },
};

const sidebarIcons: Record<string, React.ReactNode> = {
  'layout-dashboard': (<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></svg>),
  brain: (<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M12 2a7 7 0 0 1 7 7c0 2.5-1 4-2.5 5.5L15 16H9l-1.5-1.5C6 13 5 11.5 5 9a7 7 0 0 1 7-7z"/><path d="M9 16v3a2 2 0 1 0 4 0v-3"/></svg>),
  history: (<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>),
  'shield-alert': (<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>),
  activity: (<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>),
  'git-compare': (<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="18" cy="18" r="3"/><circle cx="6" cy="6" r="3"/><path d="M13 6h3a2 2 0 0 1 2 2v7"/><line x1="6" y1="9" x2="6" y2="21"/></svg>),
  'play-circle': (<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg>),
  settings: (<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>),
};

function SvgIcon({ name, size = 17 }: { name: string; size?: number }) {
  const map: Record<string, React.ReactNode> = {
    search: (<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width={size} height={size}><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>),
    bell: (<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width={size} height={size}><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/></svg>),
    refresh: (<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width={size} height={size}><path d="M23 4v6h-6"/><path d="M1 20v-6h6"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>),
    book: (<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width={size} height={size}><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>),
    x: (<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width={size} height={size}><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>),
  };
  return <>{map[name] || null}</>;
}

const TAB_KEYS: Record<string, string> = { o:'overview', b:'beliefs', t:'timeline', c:'conflicts', a:'activity', r:'compare', s:'simulator', i:'settings' };

export function Layout({
  children, activeTab, setActiveTab, tabs, sessions,
  selectedSession, setSelectedSession, loading, refreshData,
  sseConnected, trackingStatus, notif,
}: Props) {
  const meta = PAGE_META[activeTab] || PAGE_META.overview;
  const statusConnected = sseConnected && trackingStatus !== 'disconnected';
  const statusClass = statusConnected ? 'connected' : 'disconnected';
  const [searchOpen, setSearchOpen] = useState(false);
  const [notifOpen, setNotifOpen] = useState(false);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);

  const handleHelp = useCallback(() => window.open('https://AltioraLabs.github.io/beliefstate/', '_blank', 'noopener'), []);
  const pendingKey = React.useRef('');

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLSelectElement) return;
      if (e.key === '/') { e.preventDefault(); setSearchOpen(true); return; }
      if (e.key === '?') { e.preventDefault(); setShortcutsOpen(true); return; }
      if (e.key === 'g' && pendingKey.current === '') { pendingKey.current = 'g'; setTimeout(() => pendingKey.current = '', 800); return; }
      if (pendingKey.current === 'g' && TAB_KEYS[e.key]) { pendingKey.current = ''; e.preventDefault(); setActiveTab(TAB_KEYS[e.key]); return; }
      pendingKey.current = '';
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [setActiveTab]);

  return (
    <>
      <aside className="sidebar">
        <div className="sidebar-header">
          <img src="/logo.png" alt="BeliefState" className="sidebar-logo-img" />
          <span className="logo-text">BeliefState</span>
        </div>
        <nav className="sidebar-nav">
          <div className="nav-group-label">MAIN</div>
          {tabs.slice(0, 5).map(tab => (
            <button key={tab.id} className={`nav-item ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}>
              {sidebarIcons[tab.icon]}
              <span>{tab.label}</span>
            </button>
          ))}
          <div className="nav-group-label">TOOLS</div>
          {tabs.slice(5).map(tab => (
            <button key={tab.id} className={`nav-item ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}>
              {sidebarIcons[tab.icon]}
              <span>{tab.label}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-footer">
          <Select value={selectedSession || ''} onChange={v => setSelectedSession(v || null)}
            options={[{value:'',label:'Select session...'}, ...sessions.map(s => ({value:s, label:s.length>28?s.slice(0,28)+'…':s}))]}
            placeholder="Select session..." width="100%" />
          <div className="status-card">
            <div className={`status-dot ${statusClass}`} />
            <div className="status-info">
              <span className="status-label">{statusConnected ? 'Connected' : 'Disconnected'}</span>
              <span className="status-sub">{trackingStatus === 'active' ? 'Live updates' : 'Idle'}</span>
            </div>
          </div>
        </div>
      </aside>

      <div className="main-area">
        <header className="top-header">
          <div className="top-header-left">
            <h1 className="page-title">{meta.title}</h1>
            <p className="page-subtitle">{meta.subtitle}</p>
          </div>
          <div className="top-header-right">
            <button className="header-btn" onClick={() => setSearchOpen(true)} title="Search (/)">
              <SvgIcon name="search" />
            </button>
            <div className="header-btn-wrap">
              <button className="header-btn" onClick={() => setNotifOpen(o => !o)} title="Notifications">
                <SvgIcon name="bell" />
                {notif.unreadCount > 0 && <span className="notif-badge">{notif.unreadCount > 9 ? '9+' : notif.unreadCount}</span>}
              </button>
            </div>
            <button className="header-btn" onClick={handleHelp} title="Documentation">
              <SvgIcon name="book" />
            </button>
            <button className="header-btn" onClick={refreshData} disabled={loading} title="Refresh">
              <SvgIcon name="refresh" />
            </button>
          </div>
        </header>
        <main className="main-content">{children}</main>
      </div>

      {searchOpen && <GlobalSearch sessions={sessions} onNavigate={setActiveTab} onClose={() => setSearchOpen(false)} />}
      {notifOpen && <NotificationPanel notifications={notif.notifications} unreadCount={notif.unreadCount} onMarkRead={notif.markRead} onMarkAllRead={notif.markAllRead} onClear={notif.clearNotifs} onClose={() => setNotifOpen(false)} />}
      {shortcutsOpen && <KeyboardShortcuts onClose={() => setShortcutsOpen(false)} />}
    </>
  );
}
