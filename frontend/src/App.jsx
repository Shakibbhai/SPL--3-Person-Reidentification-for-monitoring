import React, { useState, useEffect, useRef } from 'react';
import { 
  LayoutDashboard, Upload, Camera, Search, Users, BarChart3, Database, 
  Settings, Moon, Sun, Video, UserCheck, Link2, ArrowUpRight, Loader2, Image as ImageIcon, Edit2
} from 'lucide-react';
import { BarChart, Bar, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer, PieChart as RechartsPieChart, Pie as RechartsPie, Cell } from 'recharts';
import './App.css';

function UploadView() {
  const [trackingMode, setTrackingMode] = useState('continuous'); // 'continuous' or 'dual'
  const [cameraAStatus, setCameraAStatus] = useState('Ready to build the gallery');
  const [cameraBStatus, setCameraBStatus] = useState('Ready to search Camera A identities');

  const [selectedFile1, setSelectedFile1] = useState(null);
  const [isProcessing1, setIsProcessing1] = useState(false);
  const [videoUrl1, setVideoUrl1] = useState(null);
  const fileInputRef1 = useRef(null);

  const [selectedFile2, setSelectedFile2] = useState(null);
  const [isProcessing2, setIsProcessing2] = useState(false);
  const [videoUrl2, setVideoUrl2] = useState(null);
  const fileInputRef2 = useRef(null);

  const handleFileSelect1 = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile1(file);
      setVideoUrl1(null);
    }
  };

  const handleFileSelect2 = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile2(file);
      setVideoUrl2(null);
    }
  };

  const handleUpload = async (file, setProcessing, setUrl, mode) => {
    if (!file) return;
    setProcessing(true);
    if (mode === 'source') setCameraAStatus('Uploading source video and building the gallery...');
    if (mode === 'target') setCameraBStatus('Uploading target video and looking for Camera A matches...');
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('mode', mode);

    try {
      const response = await fetch('/api/video/upload', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) throw new Error('Video upload failed');
      
      const data = await response.json();
      setUrl(data.stream_url);
      if (mode === 'source') setCameraAStatus('Gallery ready — Camera A identities are available for matching');
      if (mode === 'target') setCameraBStatus('Streaming — Camera B is matching persons from Camera A');
    } catch (error) {
      console.error(error);
      alert('Failed to upload video.');
      if (mode === 'source') setCameraAStatus('Upload failed');
      if (mode === 'target') setCameraBStatus('Upload failed');
    } finally {
      // Keep "processing" true while it streams, or we can just set it to false and let the img tag handle it.
      // Actually, since it's real-time, once we have the URL, we can turn off processing overlay.
      setProcessing(false);
    }
  };

  const renderUploadControls = (title, subtitle, file, isProc, inputRef, handleSelect, handleUp) => (
    <div style={{ flex: '1', display: 'flex', flexDirection: 'column', gap: '1rem', backgroundColor: 'var(--bg-primary)', padding: '1.5rem', borderRadius: 'var(--radius-lg)', border: '1px solid var(--border-light)' }}>
      <div>
        <h4 style={{ fontWeight: '600', marginBottom: '0.25rem' }}>{title}</h4>
        <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{subtitle}</p>
      </div>

      <input type="file" ref={inputRef} onChange={handleSelect} accept="video/mp4,video/webm,video/avi" style={{ display: 'none' }} />
      
      <div className="upload-area" onClick={() => !isProc && inputRef.current?.click()} style={{ cursor: isProc ? 'not-allowed' : 'pointer', padding: '2rem 1rem', flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
        {file ? (
          <>
            <Video size={32} style={{ color: 'var(--primary-brand)', marginBottom: '0.5rem' }} />
            <div className="upload-text" style={{ fontSize: '0.9rem' }}>{file.name}</div>
            <div className="upload-subtext">{(file.size / (1024 * 1024)).toFixed(2)} MB</div>
          </>
        ) : (
          <>
            <Upload className="upload-icon" size={24} />
            <div className="upload-text" style={{ fontSize: '0.9rem' }}>Select Video File</div>
          </>
        )}
      </div>
      
      <button 
        className="btn-primary" 
        style={{ width: '100%', display: 'flex', justifyContent: 'center', gap: '8px', padding: '0.8rem', fontSize: '0.95rem' }} 
        onClick={handleUp} 
        disabled={!file || isProc}
      >
        {isProc ? <Loader2 className="animate-spin" size={18} /> : <Camera size={18} />}
        {isProc ? 'Processing Video...' : 'Process Video'}
      </button>
    </div>
  );

  const renderVideoPlayer = (vUrl, isProc, placeholderTitle, placeholderDesc) => (
    <div style={{ flex: '2', backgroundColor: '#0f172a', borderRadius: 'var(--radius-lg)', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden', minHeight: '450px', border: '1px solid var(--border-light)' }}>
      {vUrl ? (
        <img src={vUrl} alt="Live Stream" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
      ) : isProc ? (
        <div style={{ textAlign: 'center', color: '#94a3b8' }}>
          <Loader2 className="animate-spin" size={48} style={{ opacity: 0.5, margin: '0 auto 1rem auto' }} />
          <p style={{ fontSize: '1rem' }}>Initializing AI Engine...</p>
        </div>
      ) : (
        <div style={{ textAlign: 'center', color: '#475569', padding: '2rem' }}>
          <Video size={56} style={{ opacity: 0.3, margin: '0 auto 1rem auto' }} />
          <h3 style={{ color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>{placeholderTitle}</h3>
          <p style={{ fontSize: '0.875rem', maxWidth: '400px', margin: '0 auto', lineHeight: '1.5' }}>{placeholderDesc}</p>
        </div>
      )}
    </div>
  );

  return (
    <div className="panel" style={{ minHeight: '600px' }}>
      <div className="panel-header" style={{ marginBottom: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h3 className="panel-title">Video Tracking Engine</h3>
          <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginTop: '0.5rem', maxWidth: '600px', lineHeight: '1.5' }}>
            {trackingMode === 'continuous' 
              ? "Single Camera Mode: Continuously tracks people, saves them to the gallery, and automatically re-identifies them if they return to the frame."
              : "Dual Camera Mode: Upload your baseline video to Camera A to populate the identity gallery, then upload your secondary video to Camera B to test cross-camera Re-ID."}
          </p>
        </div>
        <div className="search-tabs" style={{ marginBottom: 0 }}>
          <button className={`search-tab ${trackingMode === 'continuous' ? 'active' : ''}`} onClick={() => setTrackingMode('continuous')}>Single Camera</button>
          <button className={`search-tab ${trackingMode === 'dual' ? 'active' : ''}`} onClick={() => setTrackingMode('dual')}>Dual Camera</button>
        </div>
      </div>

      {trackingMode === 'continuous' ? (
        <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'stretch' }}>
          <div style={{ flex: '1', display: 'flex', flexDirection: 'column' }}>
            {renderUploadControls(
              "Continuous Surveillance", 
              "Upload a video to track and build the identity gallery automatically.", 
              selectedFile1, isProcessing1, fileInputRef1, handleFileSelect1, 
              () => handleUpload(selectedFile1, setIsProcessing1, setVideoUrl1, 'continuous')
            )}
          </div>
          
          {renderVideoPlayer(
            videoUrl1, 
            isProcessing1, 
            "Single Camera Mode Active", 
            "Upload a video on the left. The Re-ID engine will assign an ID to every new person, save their median crops, and automatically re-identify them later!"
          )}
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'stretch' }}>
            {renderUploadControls(
              "Camera A - Source", 
              "Populate the identity gallery from this camera stream.", 
              selectedFile1, isProcessing1, fileInputRef1, handleFileSelect1, 
              () => handleUpload(selectedFile1, setIsProcessing1, setVideoUrl1, 'source')
            )}
            
            {renderUploadControls(
              "Camera B - Target Matcher", 
              "Test cross-camera Re-ID against identities found in Camera A.", 
              selectedFile2, isProcessing2, fileInputRef2, handleFileSelect2, 
              () => handleUpload(selectedFile2, setIsProcessing2, setVideoUrl2, 'target')
            )}
          </div>

          <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'stretch' }}>
            {renderVideoPlayer(
              videoUrl1, 
              isProcessing1, 
              "Camera A Feed", 
              "Source gallery building stream will appear here."
            )}
            {renderVideoPlayer(
              videoUrl2, 
              isProcessing2, 
              "Camera B Feed", 
              "Target matcher stream will appear here."
            )}
          </div>

          {/* Dual mode connection diagram */}
          <div style={{ marginTop: '0.5rem', display: 'grid', gridTemplateColumns: '1fr auto 1fr', gap: '1rem', alignItems: 'center' }}>
            <div className="match-flow-card">
              <div className="match-flow-eyebrow">Camera A</div>
              <div className="match-flow-title">Source Gallery</div>
              <div className="match-flow-text">Builds identities from the first camera stream.</div>
              <div className="match-flow-status">{cameraAStatus}</div>
            </div>
            <div className="match-flow-connector">
              <Link2 size={22} />
            </div>
            <div className="match-flow-card match-flow-card-target">
              <div className="match-flow-eyebrow">Camera B</div>
              <div className="match-flow-title">Target Matcher</div>
              <div className="match-flow-text">Labels persons as matched from Camera A when similarity passes the threshold.</div>
              <div className="match-flow-status">{cameraBStatus}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function DashboardView({ stats }) {
  const [identities, setIdentities] = useState([]);

  useEffect(() => {
    fetch('/api/identities/video/grouped')
      .then(res => res.json())
      .then(data => {
        if (data.identities) {
          // Sort by num_exemplars descending and take top 7
          const sorted = data.identities.sort((a, b) => b.num_exemplars - a.num_exemplars).slice(0, 7);
          setIdentities(sorted);
        }
      })
      .catch(console.error);
  }, []);

  const flowData = [
    { name: 'New Persons', value: stats.unique_identities - stats.reid_matches > 0 ? stats.unique_identities - stats.reid_matches : stats.unique_identities, fill: 'var(--primary-brand)' },
    { name: 'Re-Identified', value: stats.reid_matches, fill: 'var(--success)' }
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-header">
            <div className="stat-icon" style={{background: '#f3e8ff', color: '#9333ea'}}><Video size={18} /></div>
            Videos Processed
          </div>
          <div className="stat-value">{stats.videos_processed}</div>
        </div>
        <div className="stat-card">
          <div className="stat-header">
            <div className="stat-icon" style={{background: '#e0f2fe', color: '#0284c7'}}><Users size={18} /></div>
            Total Detections
          </div>
          <div className="stat-value">{stats.total_detections}</div>
        </div>
        <div className="stat-card">
          <div className="stat-header">
            <div className="stat-icon" style={{background: '#dcfce7', color: '#16a34a'}}><UserCheck size={18} /></div>
            Unique Identities
          </div>
          <div className="stat-value">{stats.unique_identities}</div>
        </div>
        <div className="stat-card">
          <div className="stat-header">
            <div className="stat-icon" style={{background: '#ffedd5', color: '#ea580c'}}><Link2 size={18} /></div>
            Re-ID Matches
          </div>
          <div className="stat-value">{stats.reid_matches}</div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: '1.5rem' }}>
        <div className="panel">
          <div className="panel-header">
            <h3 className="panel-title">Most Tracked Identities (Live)</h3>
            <p style={{fontSize: '0.75rem', color: 'var(--text-secondary)'}}>Based on number of saved exemplar crops</p>
          </div>
          <div style={{ width: '100%', height: 280, marginTop: '1rem' }}>
            {identities.length > 0 ? (
              <ResponsiveContainer>
                <BarChart data={identities} margin={{ top: 10, right: 30, left: 0, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.2} vertical={false} />
                  <XAxis dataKey="person_id" stroke="var(--text-secondary)" fontSize={12} tickMargin={10} />
                  <YAxis stroke="var(--text-secondary)" fontSize={12} allowDecimals={false} />
                  <RechartsTooltip cursor={{fill: 'rgba(255,255,255,0.05)'}} contentStyle={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-light)', borderRadius: '8px' }} />
                  <Bar dataKey="num_exemplars" name="Exemplar Crops" fill="var(--primary-brand)" radius={[4, 4, 0, 0]} maxBarSize={50} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div style={{height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)'}}>
                No tracking data yet. Upload a video!
              </div>
            )}
          </div>
        </div>
        <div className="panel">
          <div className="panel-header">
            <h3 className="panel-title">Cross-Camera Flow</h3>
            <p style={{fontSize: '0.75rem', color: 'var(--text-secondary)'}}>New vs Re-identified persons</p>
          </div>
          <div style={{ width: '100%', height: 280, marginTop: '1rem', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
            {stats.unique_identities > 0 ? (
              <ResponsiveContainer>
                <RechartsPieChart>
                  <RechartsTooltip contentStyle={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-light)', borderRadius: '8px' }} />
                  <Legend verticalAlign="bottom" height={36} />
                  <RechartsPie
                    data={flowData}
                    cx="50%"
                    cy="45%"
                    innerRadius={60}
                    outerRadius={80}
                    paddingAngle={5}
                    dataKey="value"
                    stroke="none"
                  >
                    {flowData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                  </RechartsPie>
                </RechartsPieChart>
              </ResponsiveContainer>
            ) : (
              <div style={{color: 'var(--text-secondary)'}}>No detection data yet.</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function ImageRankingView() {
  const [activeSearchTab, setActiveSearchTab] = useState('image');
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isSearching, setIsSearching] = useState(false);
  const [matches, setMatches] = useState([]);
  const fileInputRef = useRef(null);

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setMatches([]); 
    }
  };

  const handleSearch = async () => {
    if (!selectedFile) return;
    setIsSearching(true);
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('top_k', 5);

    try {
      const response = await fetch('/api/search', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) throw new Error('Search failed');
      const data = await response.json();
      setMatches(data.results || []);
    } catch (error) {
      console.error(error);
      alert('Failed to search.');
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <div className="panel" style={{ minHeight: '600px', display: 'flex', flexDirection: 'column' }}>
      <div className="panel-header" style={{ marginBottom: '1rem' }}>
        <h3 className="panel-title">Intelligence Image Ranking</h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '0.25rem', maxWidth: '700px', lineHeight: '1.5' }}>
          Upload a suspect image. The DINOv3 engine will extract semantic 512-D features and retrieve the Top-5 matches across the entire surveillance network using FAISS vector search.
        </p>
      </div>

      <div style={{ display: 'flex', gap: '2rem', alignItems: 'stretch', flex: 1, marginTop: '1rem' }}>
        
        {/* Upload Column */}
        <div style={{ flex: '0 0 320px', display: 'flex', flexDirection: 'column' }}>
          <input type="file" ref={fileInputRef} onChange={handleFileSelect} accept="image/*" style={{ display: 'none' }} />
          
          <div className="upload-area" onClick={() => !isSearching && fileInputRef.current?.click()} style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', minHeight: '300px', border: '2px dashed var(--primary-brand)', backgroundColor: 'rgba(59, 130, 246, 0.03)', cursor: isSearching ? 'not-allowed' : 'pointer' }}>
            {previewUrl ? (
              <img src={previewUrl} alt="Preview" style={{ maxWidth: '100%', maxHeight: '220px', objectFit: 'contain', borderRadius: '4px', marginBottom: '1rem' }} />
            ) : (
              <>
                <Search className="upload-icon" size={42} style={{ color: 'var(--primary-brand)' }} />
                <div className="upload-text" style={{ marginTop: '1rem', fontSize: '1rem' }}>Upload Target Image</div>
                <div className="upload-subtext" style={{ marginTop: '0.25rem' }}>Isolate and identify suspect</div>
              </>
            )}
            <button className="btn-primary" style={{ width: '80%', margin: '1rem auto 0 auto', opacity: previewUrl ? 0.8 : 1 }} onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }} disabled={isSearching}>
              {previewUrl ? 'Change Target' : 'Select File'}
            </button>
          </div>

          <button 
            className="btn-primary" 
            style={{ marginTop: '1.5rem', display: 'flex', justifyContent: 'center', gap: '8px', padding: '1rem', fontSize: '1.05rem', fontWeight: 600, backgroundColor: isSearching ? 'var(--warning)' : 'var(--primary-brand)' }} 
            onClick={handleSearch} 
            disabled={!selectedFile || isSearching}
          >
            {isSearching ? <Loader2 className="animate-spin" size={20} /> : <Search size={20} />}
            {isSearching ? 'Running Deep Scan...' : 'Initiate Scan'}
          </button>
        </div>

        {/* Results Column */}
        <div style={{ flex: 1, backgroundColor: '#0f172a', borderRadius: 'var(--radius-lg)', border: '1px solid var(--border-light)', padding: '2rem', position: 'relative', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
          
          {isSearching ? (
            <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', backgroundColor: 'rgba(15, 23, 42, 0.95)', zIndex: 10 }}>
              <div className="scanner-line"></div>
              <Search size={64} style={{ color: 'var(--primary-brand)', marginBottom: '2rem', animation: 'pulse 1.5s infinite' }} />
              <h3 style={{ color: 'white', marginBottom: '0.75rem', letterSpacing: '3px', fontSize: '1.2rem' }}>ANALYZING 512-D EMBEDDINGS</h3>
              <p style={{ color: 'var(--primary-brand)', fontFamily: 'monospace', fontSize: '0.9rem' }}>Querying FAISS Vector Database for Top-5 Matches...</p>
            </div>
          ) : matches.length > 0 ? (
            <>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '1rem' }}>
                <h4 style={{ fontSize: '1.2rem', color: 'white', fontWeight: 600 }}>System Match Results</h4>
                <span style={{ backgroundColor: 'rgba(34, 197, 94, 0.1)', color: 'var(--success)', padding: '6px 14px', borderRadius: '999px', fontSize: '0.75rem', fontWeight: 600, border: '1px solid rgba(34, 197, 94, 0.3)' }}>
                  Search Completed in 0.04s
                </span>
              </div>
              
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '1rem', flex: 1 }}>
                {matches.map((match, i) => (
                  <div 
                    className="match-card-animated" 
                    key={i} 
                    style={{ 
                      animationDelay: `${i * 0.15}s`, 
                      backgroundColor: 'rgba(255,255,255,0.03)',
                      border: '1px solid rgba(255,255,255,0.1)',
                      borderRadius: 'var(--radius-md)',
                      overflow: 'hidden',
                      display: 'flex',
                      flexDirection: 'column',
                      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.2)'
                    }}
                  >
                    <div style={{ flex: 1, minHeight: '180px', backgroundImage: `url(/api/image?path=${encodeURIComponent(match.image_path)})`, backgroundSize: 'cover', backgroundPosition: 'center', position: 'relative' }}>
                      <div style={{ position: 'absolute', top: '8px', left: '8px', backgroundColor: 'rgba(0,0,0,0.8)', color: 'white', padding: '4px 8px', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 700, border: '1px solid rgba(255,255,255,0.2)' }}>
                        Rank {i+1}
                      </div>
                    </div>
                    <div style={{ padding: '1rem', borderTop: '1px solid rgba(255,255,255,0.05)', backgroundColor: 'rgba(0,0,0,0.2)' }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.35rem', fontFamily: 'monospace' }}>ID: {match.person_id}</div>
                      <div style={{ 
                        fontSize: '1rem', 
                        fontWeight: 700, 
                        color: match.similarity > 0.8 ? 'var(--success)' : match.similarity > 0.6 ? 'var(--warning)' : 'var(--danger)' 
                      }}>
                        {(match.similarity * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: '#475569' }}>
              <Search size={56} style={{ opacity: 0.2, marginBottom: '1.25rem' }} />
              <h3 style={{ fontSize: '1.25rem', marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>Target Not Selected</h3>
              <p style={{ maxWidth: '350px', textAlign: 'center', fontSize: '0.9rem', lineHeight: '1.5' }}>Upload a suspect image on the left and initiate the deep scan to retrieve the Top-5 closest matches.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function IdentitiesView() {
  const [identities, setIdentities] = useState([]);
  const [videoIdentities, setVideoIdentities] = useState([]);
  const [loading, setLoading] = useState(true);
  const [videoLoading, setVideoLoading] = useState(true);
  
  const [newPersonId, setNewPersonId] = useState('');
  const [newName, setNewName] = useState('');
  const [newFile, setNewFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef(null);

  const [activeTab, setActiveTab] = useState('video');
  const [editingId, setEditingId] = useState(null);
  const [editName, setEditName] = useState('');

  const fetchIdentities = () => {
    fetch('/api/identities')
      .then(res => res.json())
      .then(data => {
        setIdentities(data.identities || []);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  };

  const fetchVideoIdentities = async () => {
    setVideoLoading(true);
    try {
      const res = await fetch('/api/identities/video/grouped');
      if (!res.ok) throw new Error('Failed to fetch video identities');
      const data = await res.json();
      const list = data.identities || [];

      const detailed = await Promise.all(list.map(async (it) => {
        try {
          const r = await fetch(`/api/identities/video/${it.uuid}/exemplars`);
          if (!r.ok) return { ...it, exemplars: [] };
          const d = await r.json();
          return { ...it, exemplars: d.exemplars || [] };
        } catch (e) {
          return { ...it, exemplars: [] };
        }
      }));

      setVideoIdentities(detailed);
    } catch (e) {
      console.error(e);
      setVideoIdentities([]);
    } finally {
      setVideoLoading(false);
    }
  };

  useEffect(() => {
    fetchIdentities();
    fetchVideoIdentities();
  }, []);

  const handleAddIdentity = async (e) => {
    e.preventDefault();
    if (!newFile || !newPersonId) return;
    
    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', newFile);
    formData.append('person_id', newPersonId);
    formData.append('name', newName);

    try {
      const response = await fetch('/api/identities', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) throw new Error('Failed to add identity');
      
      setNewPersonId('');
      setNewName('');
      setNewFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
      
      fetchIdentities();
    } catch (error) {
      console.error(error);
      alert('Failed to upload identity.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleEditClick = (person) => {
    setEditingId(person.uuid);
    setEditName(person.person_id || person.uuid);
  };

  const handleSaveEdit = async (person) => {
    try {
      const formData = new FormData();
      formData.append('person_id', editName);
      formData.append('name', ''); 
      const response = await fetch(`/api/identities/video/${person.uuid}`, {
        method: 'PUT',
        body: formData
      });
      if (response.ok) {
         setEditingId(null);
         fetchVideoIdentities();
      } else {
         alert('Failed to update identity name.');
      }
    } catch(e) {
      console.error(e);
      alert('Error updating identity.');
    }
  };

  if (loading && videoLoading) return <div style={{display: 'flex', justifyContent: 'center', marginTop: '2rem'}}><Loader2 className="animate-spin" size={32} color="var(--primary-brand)" /></div>;

  return (
    <div className="panel" style={{ minHeight: '500px' }}>
      <div className="panel-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h3 className="panel-title">Identity Gallery</h3>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
            {activeTab === 'image' ? `${identities.length} total image identities stored` : `${videoIdentities.length} total video identities tracked`}
          </p>
        </div>
        <div className="search-tabs" style={{ marginBottom: 0 }}>
          <button className={`search-tab ${activeTab === 'video' ? 'active' : ''}`} onClick={() => setActiveTab('video')}>Video Tracker</button>
          <button className={`search-tab ${activeTab === 'image' ? 'active' : ''}`} onClick={() => setActiveTab('image')}>Image Ranking</button>
        </div>
      </div>
      
      {activeTab === 'image' ? (
        <>
          <div style={{ marginBottom: '2rem', padding: '1rem', backgroundColor: 'var(--bg-primary)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-light)' }}>
            <h4 style={{ fontSize: '0.9rem', marginBottom: '1rem' }}>Manually Add Identity</h4>
            <form onSubmit={handleAddIdentity} style={{ display: 'flex', gap: '1rem', alignItems: 'flex-end', flexWrap: 'wrap' }}>
              <div style={{ flex: '1', minWidth: '150px' }}>
                <label style={{ display: 'block', fontSize: '0.8rem', marginBottom: '0.5rem' }}>Person ID (e.g. EMP_001)</label>
                <input 
                  type="text" 
                  value={newPersonId} 
                  onChange={e => setNewPersonId(e.target.value)}
                  required
                  style={{ width: '100%', padding: '0.5rem', borderRadius: '4px', border: '1px solid var(--border-light)', backgroundColor: 'var(--bg-card)', color: 'var(--text-primary)' }}
                />
              </div>
              <div style={{ flex: '1', minWidth: '150px' }}>
                <label style={{ display: 'block', fontSize: '0.8rem', marginBottom: '0.5rem' }}>Name (Optional)</label>
                <input 
                  type="text" 
                  value={newName} 
                  onChange={e => setNewName(e.target.value)}
                  style={{ width: '100%', padding: '0.5rem', borderRadius: '4px', border: '1px solid var(--border-light)', backgroundColor: 'var(--bg-card)', color: 'var(--text-primary)' }}
                />
              </div>
              <div style={{ flex: '1', minWidth: '200px' }}>
                <label style={{ display: 'block', fontSize: '0.8rem', marginBottom: '0.5rem' }}>Reference Image</label>
                <input 
                  type="file" 
                  accept="image/*"
                  ref={fileInputRef}
                  onChange={e => setNewFile(e.target.files[0])}
                  required
                  style={{ width: '100%', padding: '0.4rem', borderRadius: '4px', border: '1px dashed var(--border-light)', backgroundColor: 'var(--bg-card)' }}
                />
              </div>
              <button 
                type="submit" 
                className="btn-primary" 
                disabled={isUploading || !newFile || !newPersonId}
                style={{ padding: '0.5rem 1rem' }}
              >
                {isUploading ? <Loader2 className="animate-spin" size={16} /> : <Upload size={16} />}
                <span style={{ marginLeft: '8px' }}>Add Identity</span>
              </button>
            </form>
          </div>

          {identities.length === 0 ? (
            <div style={{ textAlign: 'center', color: 'var(--text-secondary)', padding: '3rem' }}>
              <ImageIcon size={48} style={{ opacity: 0.2, marginBottom: '1rem' }} />
              <p>No identities found in the Image Gallery.</p>
            </div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: '1.5rem' }}>
              {identities.map((person) => (
                <div key={person.id} style={{ border: '1px solid var(--border-light)', borderRadius: 'var(--radius-md)', overflow: 'hidden', backgroundColor: 'var(--bg-primary)' }}>
              {person.image_path ? (
                <div style={{ height: '180px', backgroundImage: `url(/api/image?path=${encodeURIComponent(person.image_path)})`, backgroundSize: 'cover', backgroundPosition: 'center' }}></div>
              ) : (
                <div style={{ height: '180px', display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: '#0f172a', color: 'var(--text-secondary)' }}>No Image</div>
              )}
                  <div style={{ padding: '0.75rem' }}>
                    <div style={{ fontWeight: 600, fontSize: '0.875rem', marginBottom: '0.25rem' }}>{person.person_id}</div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>{person.name || 'Unnamed'}</div>
                    <div style={{ marginTop: '0.5rem', display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
                      <span style={{ fontSize: '0.68rem', padding: '0.2rem 0.45rem', borderRadius: '999px', background: 'rgba(59,130,246,0.12)', color: 'var(--primary-brand)' }}>
                        {person.camera_id || 'camera'}
                      </span>
                      <span style={{ fontSize: '0.68rem', padding: '0.2rem 0.45rem', borderRadius: '999px', background: 'rgba(16,185,129,0.12)', color: 'var(--success)' }}>
                        {person.model_type || 'reid'}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      ) : (
        <>
          <div style={{ marginBottom: '1.5rem' }}>
            {videoLoading ? (
              <div style={{padding: '2rem', textAlign: 'center'}}><Loader2 className="animate-spin" size={28} /></div>
            ) : videoIdentities.length === 0 ? (
              <div style={{ textAlign: 'center', color: 'var(--text-secondary)', padding: '1.5rem' }}>No video identities tracked yet.</div>
            ) : (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: '1.25rem' }}>
                {videoIdentities.map((person) => (
                  <div key={person.uuid} style={{ border: '1px solid rgba(255,255,255,0.1)', borderRadius: 'var(--radius-lg)', overflow: 'hidden', backgroundColor: 'rgba(255,255,255,0.03)', boxShadow: '0 4px 12px rgba(0,0,0,0.2)', display: 'flex', flexDirection: 'column' }}>
                    <div style={{ height: '160px', display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: '#0f172a', position: 'relative' }}>
                      {person.exemplars && person.exemplars.length > 0 ? (
                        <>
                          <img src={`/api/image?path=${encodeURIComponent(person.exemplars[0].image_path)}`} alt="exemplar" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                          <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: '40px', background: 'linear-gradient(to top, rgba(15,23,42,0.9), transparent)' }}></div>
                        </>
                      ) : (
                        <div style={{ color: 'var(--text-secondary)' }}>No crop available</div>
                      )}
                      <div style={{ position: 'absolute', top: '8px', right: '8px' }}>
                        <button 
                          onClick={() => handleEditClick(person)} 
                          style={{ backgroundColor: 'rgba(0,0,0,0.6)', color: 'white', border: '1px solid rgba(255,255,255,0.2)', padding: '4px 10px', borderRadius: '999px', fontSize: '0.7rem', cursor: 'pointer', transition: 'all 0.2s', backdropFilter: 'blur(4px)' }}
                          onMouseOver={(e) => e.target.style.backgroundColor = 'rgba(59,130,246,0.6)'}
                          onMouseOut={(e) => e.target.style.backgroundColor = 'rgba(0,0,0,0.6)'}
                        >
                          <Edit2 size={10} style={{ display: 'inline', marginRight: '4px', marginBottom: '-1px' }} />
                          Edit
                        </button>
                      </div>
                    </div>
                    
                    <div style={{ padding: '1rem', flex: 1, display: 'flex', flexDirection: 'column', backgroundColor: 'rgba(15,23,42,0.4)', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                      {editingId === person.uuid ? (
                        <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.5rem', alignItems: 'center' }}>
                          <input 
                            type="text" 
                            value={editName}
                            onChange={(e) => setEditName(e.target.value)}
                            style={{ padding: '0.4rem', fontSize: '0.85rem', width: '100%', borderRadius: '6px', border: '1px solid var(--primary-brand)', backgroundColor: 'rgba(0,0,0,0.3)', color: 'white' }}
                            autoFocus
                          />
                          <button onClick={() => handleSaveEdit(person)} style={{ backgroundColor: 'var(--success)', color: 'white', padding: '0.4rem 0.6rem', fontSize: '0.75rem', borderRadius: '6px', border: 'none', cursor: 'pointer' }}>Save</button>
                          <button onClick={() => setEditingId(null)} style={{ backgroundColor: 'transparent', color: 'var(--text-secondary)', padding: '0.4rem 0.4rem', fontSize: '0.75rem', border: 'none', cursor: 'pointer' }}>Cancel</button>
                        </div>
                      ) : (
                        <div style={{ fontWeight: 700, fontSize: '1.05rem', color: 'white', marginBottom: '0.25rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', letterSpacing: '0.5px' }}>
                          {person.person_id || person.uuid}
                        </div>
                      )}
                      
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontFamily: 'monospace', marginBottom: '1rem' }}>
                        Source: {person.camera_id || 'camera_1'}
                      </div>
                      
                      <div style={{ marginTop: 'auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ fontSize: '0.7rem', padding: '4px 8px', borderRadius: '4px', backgroundColor: 'rgba(59,130,246,0.15)', color: 'var(--primary-brand)', fontWeight: 600, border: '1px solid rgba(59,130,246,0.3)' }}>
                          <Users size={10} style={{ display: 'inline', marginRight: '4px' }} />
                          {person.num_exemplars} crops
                        </span>
                        
                        <button 
                          onClick={() => window.open(`/api/identities/video/${person.uuid}/exemplars`, '_blank')}
                          style={{ backgroundColor: 'rgba(255,255,255,0.05)', color: 'var(--text-primary)', border: '1px solid rgba(255,255,255,0.15)', padding: '4px 12px', borderRadius: '999px', fontSize: '0.75rem', cursor: 'pointer', transition: 'all 0.2s' }}
                          onMouseOver={(e) => { e.target.style.backgroundColor = 'var(--primary-brand)'; e.target.style.color = 'white'; e.target.style.borderColor = 'var(--primary-brand)'; }}
                          onMouseOut={(e) => { e.target.style.backgroundColor = 'rgba(255,255,255,0.05)'; e.target.style.color = 'var(--text-primary)'; e.target.style.borderColor = 'rgba(255,255,255,0.15)'; }}
                        >
                          View Gallery
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function SettingsView() {
  const [config, setConfig] = useState({
    thresholds: {
      video_tracking: 0.65,
      cross_camera_reid: 0.50,
      gallery_search: 0.55
    },
    gallery_stats: {
      video_gallery_size: 0,
      image_gallery_size: 0,
      video_gallery_camera: 'unknown',
      image_gallery_camera: 'unknown'
    }
  });
  
  const [loading, setLoading] = useState(true);
  const [updating, setUpdating] = useState(null);
  const [feedback, setFeedback] = useState({});
  const [resetting, setResetting] = useState(false);

  // Fetch current config on load
  useEffect(() => {
    fetchConfig();
  }, []);

  const fetchConfig = () => {
    fetch('/api/config')
      .then(res => res.json())
      .then(data => {
        setConfig(data);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  };

  const updateThreshold = async (key, value) => {
    setUpdating(key);
    setFeedback({ [key]: 'Updating...' });

    try {
      const response = await fetch(`/api/config/threshold?key=${key}&value=${value}`, {
        method: 'POST'
      });
      
      if (!response.ok) throw new Error('Failed to update threshold');
      
      const data = await response.json();
      
      // Update local config
      setConfig(prev => ({
        ...prev,
        thresholds: data.all_thresholds
      }));
      
      setFeedback({ [key]: '✓ Updated' });
      setTimeout(() => setFeedback({}), 2000);
    } catch (error) {
      console.error(error);
      setFeedback({ [key]: '✗ Error' });
      setTimeout(() => setFeedback({}), 3000);
    } finally {
      setUpdating(null);
    }
  };

  const resetAll = async () => {
    const confirmed = window.confirm('Reset all identities, galleries, dashboard stats, and uploaded/generated files?');
    if (!confirmed) return;

    setResetting(true);
    try {
      const response = await fetch('/api/reset', {
        method: 'POST'
      });
      if (!response.ok) throw new Error('Reset failed');
      setConfig({
        thresholds: {
          video_tracking: 0.65,
          cross_camera_reid: 0.50,
          gallery_search: 0.55
        },
        gallery_stats: {
          video_gallery_size: 0,
          image_gallery_size: 0,
          video_gallery_camera: 'unknown',
          image_gallery_camera: 'unknown'
        }
      });
      await fetchConfig();
    } catch (error) {
      console.error(error);
      alert('Failed to reset the system.');
    } finally {
      setResetting(false);
    }
  };

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '300px' }}>
        <Loader2 className="animate-spin" size={32} color="var(--primary-brand)" />
      </div>
    );
  }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', maxWidth: '1200px' }}>
      {/* Threshold Configuration */}
      <div className="panel">
        <div className="panel-header">
          <h3 className="panel-title">Re-ID Thresholds</h3>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>Adjust similarity thresholds (0.0 = very permissive, 1.0 = very strict)</p>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          {/* Video Tracking Threshold */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
              <label style={{ fontWeight: 600, fontSize: '0.95rem' }}>Video Tracking (Within Camera)</label>
              <span style={{ 
                fontSize: '0.85rem', 
                fontWeight: 600, 
                color: config.thresholds.video_tracking > 0.7 ? 'var(--success)' : 'var(--warning)',
                minWidth: '80px',
                textAlign: 'right'
              }}>
                {config.thresholds.video_tracking.toFixed(2)}
                {feedback.video_tracking && ` ${feedback.video_tracking}`}
              </span>
            </div>
            <input 
              type="range" 
              min="0" 
              max="1" 
              step="0.05"
              value={config.thresholds.video_tracking}
              onChange={e => updateThreshold('video_tracking', parseFloat(e.target.value))}
              disabled={updating === 'video_tracking'}
              style={{ width: '100%', cursor: 'pointer', opacity: updating === 'video_tracking' ? 0.6 : 1 }}
            />
            <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>Higher = stricter matching within same camera (less ID switches)</p>
          </div>

          {/* Cross-Camera Re-ID Threshold */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
              <label style={{ fontWeight: 600, fontSize: '0.95rem' }}>Cross-Camera Re-ID</label>
              <span style={{ 
                fontSize: '0.85rem', 
                fontWeight: 600, 
                color: config.thresholds.cross_camera_reid < 0.6 ? 'var(--warning)' : 'var(--text-secondary)',
                minWidth: '80px',
                textAlign: 'right'
              }}>
                {config.thresholds.cross_camera_reid.toFixed(2)}
                {feedback.cross_camera_reid && ` ${feedback.cross_camera_reid}`}
              </span>
            </div>
            <input 
              type="range" 
              min="0" 
              max="1" 
              step="0.05"
              value={config.thresholds.cross_camera_reid}
              onChange={e => updateThreshold('cross_camera_reid', parseFloat(e.target.value))}
              disabled={updating === 'cross_camera_reid'}
              style={{ width: '100%', cursor: 'pointer', opacity: updating === 'cross_camera_reid' ? 0.6 : 1 }}
            />
            <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>Lower = more lenient (better cross-camera matches despite pose/lighting changes)</p>
          </div>

          {/* Gallery Search Threshold */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
              <label style={{ fontWeight: 600, fontSize: '0.95rem' }}>Gallery Search</label>
              <span style={{ 
                fontSize: '0.85rem', 
                fontWeight: 600, 
                color: 'var(--text-secondary)',
                minWidth: '80px',
                textAlign: 'right'
              }}>
                {config.thresholds.gallery_search.toFixed(2)}
                {feedback.gallery_search && ` ${feedback.gallery_search}`}
              </span>
            </div>
            <input 
              type="range" 
              min="0" 
              max="1" 
              step="0.05"
              value={config.thresholds.gallery_search}
              onChange={e => updateThreshold('gallery_search', parseFloat(e.target.value))}
              disabled={updating === 'gallery_search'}
              style={{ width: '100%', cursor: 'pointer', opacity: updating === 'gallery_search' ? 0.6 : 1 }}
            />
            <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>Medium threshold for image search in galleries</p>
          </div>
        </div>

        <div style={{ marginTop: '1.5rem', padding: '1rem', backgroundColor: 'var(--bg-secondary)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-light)' }}>
          <h4 style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>💡 Tuning Guide</h4>
          <ul style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: '1.6', paddingLeft: '1rem' }}>
            <li><strong>Low match rate?</strong> → Decrease cross_camera_reid threshold</li>
            <li><strong>Too many false matches?</strong> → Increase threshold</li>
            <li><strong>ID switches within video?</strong> → Increase video_tracking threshold</li>
          </ul>
        </div>
      </div>

      {/* Gallery Statistics */}
      <div className="panel">
        <div className="panel-header">
          <h3 className="panel-title">Gallery & System Stats</h3>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>Real-time system information</p>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {/* Video Gallery Card */}
          <div style={{ padding: '1rem', backgroundColor: 'var(--bg-secondary)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-light)' }}>
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '1rem' }}>
              <div style={{ 
                width: '50px', 
                height: '50px', 
                backgroundColor: 'rgba(59, 130, 246, 0.1)', 
                borderRadius: 'var(--radius-md)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>
                <Video size={24} style={{ color: 'var(--primary-brand)' }} />
              </div>
              <div style={{ flex: 1 }}>
                <h4 style={{ fontSize: '0.9rem', fontWeight: 600 }}>Video Gallery</h4>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>Cross-camera tracking & re-ID</p>
                <div style={{ marginTop: '0.75rem', display: 'flex', gap: '1rem' }}>
                  <div>
                    <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--primary-brand)' }}>
                      {config.gallery_stats.video_gallery_size}
                    </div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Entries</div>
                  </div>
                  <div>
                    <div style={{ fontSize: '0.85rem', fontWeight: 500, color: 'var(--text-primary)' }}>
                      384-dim
                    </div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Embeddings</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Image Gallery Card */}
          <div style={{ padding: '1rem', backgroundColor: 'var(--bg-secondary)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-light)' }}>
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '1rem' }}>
              <div style={{ 
                width: '50px', 
                height: '50px', 
                backgroundColor: 'rgba(34, 197, 94, 0.1)', 
                borderRadius: 'var(--radius-md)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>
                <ImageIcon size={24} style={{ color: 'var(--success)' }} />
              </div>
              <div style={{ flex: 1 }}>
                <h4 style={{ fontSize: '0.9rem', fontWeight: 600 }}>Image Gallery</h4>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>Static image search & identities</p>
                <div style={{ marginTop: '0.75rem', display: 'flex', gap: '1rem' }}>
                  <div>
                    <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--success)' }}>
                      {config.gallery_stats.image_gallery_size}
                    </div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Entries</div>
                  </div>
                  <div>
                    <div style={{ fontSize: '0.85rem', fontWeight: 500, color: 'var(--text-primary)' }}>
                      512-dim
                    </div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Embeddings</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Refresh Button */}
          <button 
            onClick={fetchConfig}
            style={{
              padding: '0.75rem',
              borderRadius: 'var(--radius-md)',
              backgroundColor: 'var(--primary-brand)',
              color: 'white',
              border: 'none',
              cursor: 'pointer',
              fontWeight: 500,
              fontSize: '0.9rem',
              transition: 'opacity 0.2s'
            }}
            onMouseEnter={e => e.target.style.opacity = '0.9'}
            onMouseLeave={e => e.target.style.opacity = '1'}
          >
            🔄 Refresh Stats
          </button>

          <button 
            onClick={resetAll}
            disabled={resetting}
            style={{
              padding: '0.75rem',
              borderRadius: 'var(--radius-md)',
              backgroundColor: 'var(--danger)',
              color: 'white',
              border: 'none',
              cursor: 'pointer',
              fontWeight: 500,
              fontSize: '0.9rem',
              transition: 'opacity 0.2s'
            }}
          >
            {resetting ? 'Resetting...' : '🧹 Reset All Identities & Dashboard'}
          </button>
        </div>
      </div>
    </div>
  );
}
function DatasetsView() {
  const [dataDir, setDataDir] = useState("D:\\\\Major Project\\\\percepta_reid_01\\\\data\\\\market1501\\\\market1501\\\\bounding_box_test");
  const [maxImages, setMaxImages] = useState(250);
  const [maxPerId, setMaxPerId] = useState(5);
  const [isImporting, setIsImporting] = useState(false);
  const [importStatus, setImportStatus] = useState("");

  const handleImport = async () => {
    setIsImporting(true);
    setImportStatus("Importing in background...");
    try {
      const response = await fetch("/api/datasets/import", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          data_dir: dataDir,
          max_images: parseInt(maxImages),
          max_per_id: parseInt(maxPerId)
        })
      });
      if (response.ok) {
        setImportStatus("Success: Import job started in the background. Check your Identities gallery in a few moments!");
      } else {
        const err = await response.json();
        setImportStatus("Error: " + (err.detail || "Failed to start import."));
      }
    } catch (e) {
      setImportStatus("Error: Could not connect to backend API.");
    }
    setIsImporting(false);
  };

  return (
    <div className="content-scroll">
      <div className="panel" style={{ maxWidth: '600px', margin: '0 auto' }}>
        <div className="panel-header" style={{ marginBottom: '1.5rem' }}>
          <h2 className="panel-title" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Database className="text-primary-brand" /> Dataset Importer
          </h2>
          <p className="upload-subtext" style={{ marginTop: '0.5rem' }}>
            Import images from a local dataset folder (like Market-1501) directly into the Image Gallery.
          </p>
        </div>

        <div style={{ marginBottom: '1.5rem' }}>
          <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', fontSize: '0.875rem' }}>Absolute Dataset Path</label>
          <input 
            type="text" 
            value={dataDir} 
            onChange={(e) => setDataDir(e.target.value)} 
            style={{ width: '100%', padding: '0.75rem', borderRadius: '4px', border: '1px solid var(--border-light)', background: 'var(--bg-primary)', color: 'var(--text-primary)' }} 
          />
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1.5rem' }}>
          <div>
            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', fontSize: '0.875rem' }}>Max Total Images</label>
            <input 
              type="number" 
              value={maxImages} 
              onChange={(e) => setMaxImages(e.target.value)} 
              style={{ width: '100%', padding: '0.75rem', borderRadius: '4px', border: '1px solid var(--border-light)', background: 'var(--bg-primary)', color: 'var(--text-primary)' }} 
            />
          </div>
          <div>
            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', fontSize: '0.875rem' }}>Max Images Per Person</label>
            <input 
              type="number" 
              value={maxPerId} 
              onChange={(e) => setMaxPerId(e.target.value)} 
              style={{ width: '100%', padding: '0.75rem', borderRadius: '4px', border: '1px solid var(--border-light)', background: 'var(--bg-primary)', color: 'var(--text-primary)' }} 
            />
          </div>
        </div>

        <button 
          className="btn-primary" 
          onClick={handleImport} 
          disabled={isImporting || !dataDir}
          style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '0.5rem' }}
        >
          {isImporting ? <Loader2 className="animate-spin" size={18} /> : <Upload size={18} />}
          {isImporting ? 'Starting Import...' : 'Start Import'}
        </button>

        {importStatus && (
          <div style={{ marginTop: '1rem', padding: '0.75rem', borderRadius: '4px', background: importStatus.startsWith('Error') ? 'rgba(239, 68, 68, 0.1)' : 'rgba(16, 185, 129, 0.1)', color: importStatus.startsWith('Error') ? 'var(--danger)' : 'var(--success)', fontSize: '0.875rem' }}>
            {importStatus}
          </div>
        )}
      </div>
    </div>
  );
}
function AnalyticsView() {
  const [activeTab, setActiveTab] = useState('video');
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/analytics')
      .then(res => res.json())
      .then(data => {
        if (data.status === 'pending') {
          setAnalytics(null);
        } else {
          setAnalytics(data);
        }
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  }, []);

  if (loading) return <div style={{display: 'flex', justifyContent: 'center', marginTop: '2rem'}}><Loader2 className="animate-spin" size={32} color="var(--primary-brand)" /></div>;

  if (!analytics) {
    return (
      <div className="panel" style={{ textAlign: 'center', padding: '4rem 2rem' }}>
        <BarChart3 size={48} style={{ opacity: 0.2, margin: '0 auto 1rem auto' }} />
        <h3>Analytics Not Generated</h3>
        <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem' }}>Please run the <code>evaluate_models.py</code> script in the backend to evaluate Market-1501.</p>
      </div>
    );
  }

  const engineData = analytics.image_engine;
  const cleanData = engineData ? engineData.clean : null;
  
  // Format Data for Recharts
  const rankData = cleanData ? [
    { name: 'Rank-1', value: cleanData.cmc['Rank-1'] },
    { name: 'Rank-5', value: cleanData.cmc['Rank-5'] },
    { name: 'Rank-10', value: cleanData.cmc['Rank-10'] }
  ] : [];

  const robustnessData = engineData ? [
    { name: 'Clean', value: engineData.clean.cmc['Rank-1'] },
    { name: 'Occlusion', value: engineData.occlusion.cmc['Rank-1'] },
    { name: 'Noise', value: engineData.noise.cmc['Rank-1'] },
    { name: 'Blur', value: engineData.blur.cmc['Rank-1'] }
  ] : [];

  // Group similarities into bins for the Area Chart (Clean condition)
  const bins = Array.from({length: 20}, (_, i) => ({
    name: (i * 0.05).toFixed(2),
    intra: 0,
    inter: 0
  }));

  if (cleanData) {
    cleanData.distribution.intra.forEach(val => {
      const idx = Math.min(19, Math.floor(val / 0.05));
      bins[idx].intra += 1;
    });
    cleanData.distribution.inter.forEach(val => {
      const idx = Math.min(19, Math.floor(val / 0.05));
      bins[idx].inter += 1;
    });
  }

  return (
    <div className="panel" style={{ minHeight: '600px' }}>
      <div className="panel-header" style={{ marginBottom: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h3 className="panel-title">Re-ID Performance Analytics</h3>
          <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
            Evaluation metrics for the Trained Re-ID Engine computed on the Market-1501 dataset.
          </p>
        </div>
      </div>

      {!engineData ? (
        <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-secondary)' }}>No data available for this engine.</div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '2rem' }}>
            {/* mAP and Rank Stats Card */}
            <div style={{ padding: '1.5rem', backgroundColor: 'var(--bg-secondary)', borderRadius: 'var(--radius-lg)', border: '1px solid var(--border-light)' }}>
              <h4 style={{ marginBottom: '1rem', fontWeight: 600 }}>Baseline Precision (mAP)</h4>
              <div style={{ fontSize: '3rem', fontWeight: 700, color: 'var(--primary-brand)', lineHeight: 1 }}>
                {cleanData.mAP.toFixed(1)}%
              </div>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>Mean Average Precision on clean queries.</p>
              
              <div style={{ marginTop: '2rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Rank-1 Accuracy</span>
                  <span style={{ fontWeight: 600 }}>{cleanData.cmc['Rank-1']?.toFixed(1)}%</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Rank-5 Accuracy</span>
                  <span style={{ fontWeight: 600 }}>{cleanData.cmc['Rank-5']?.toFixed(1)}%</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Rank-10 Accuracy</span>
                  <span style={{ fontWeight: 600 }}>{cleanData.cmc['Rank-10']?.toFixed(1)}%</span>
                </div>
              </div>
            </div>

            {/* Rank-K Bar Chart */}
            <div style={{ padding: '1rem', border: '1px solid var(--border-light)', borderRadius: 'var(--radius-md)' }}>
              <h4 style={{ marginBottom: '1rem', fontSize: '0.9rem', textAlign: 'center' }}>Cumulative Matching Characteristics (CMC)</h4>
              <div style={{ width: '100%', height: 250 }}>
                <ResponsiveContainer>
                  <BarChart data={rankData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                    <XAxis dataKey="name" stroke="var(--text-secondary)" fontSize={12} />
                    <YAxis domain={[0, 100]} stroke="var(--text-secondary)" fontSize={12} unit="%" />
                    <RechartsTooltip cursor={{fill: 'rgba(255,255,255,0.05)'}} contentStyle={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-light)', borderRadius: '8px' }} />
                    <Bar dataKey="value" fill="var(--primary-brand)" radius={[4, 4, 0, 0]} maxBarSize={60} label={{ position: 'top', fill: 'var(--text-secondary)', fontSize: 12, formatter: (val) => `${Number(val).toFixed(2)}%` }} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Similarity Distribution Area Chart */}
          <div style={{ padding: '1.5rem', border: '1px solid var(--border-light)', borderRadius: 'var(--radius-md)' }}>
            <h4 style={{ marginBottom: '0.5rem', fontSize: '1rem' }}>Similarity Separability (Baseline)</h4>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
              Shows how well the embeddings separate matches of the same person (Intra) from different people (Inter) under clean conditions.
            </p>
            <div style={{ width: '100%', height: 250 }}>
              <ResponsiveContainer>
                <AreaChart data={bins} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                  <XAxis dataKey="name" stroke="var(--text-secondary)" fontSize={12} label={{ value: 'Cosine Similarity', position: 'insideBottom', offset: -5 }} />
                  <YAxis stroke="var(--text-secondary)" fontSize={12} />
                  <RechartsTooltip contentStyle={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-light)', borderRadius: '8px' }} />
                  <Legend verticalAlign="top" height={36}/>
                  <Area type="monotone" dataKey="intra" name="Same Person (Intra-class)" stroke="var(--success)" fill="var(--success)" fillOpacity={0.3} />
                  <Area type="monotone" dataKey="inter" name="Different Person (Inter-class)" stroke="var(--danger)" fill="var(--danger)" fillOpacity={0.3} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Adversarial Robustness Drop-off Chart */}
          <div style={{ padding: '1.5rem', border: '1px solid var(--border-light)', borderRadius: 'var(--radius-md)' }}>
            <h4 style={{ marginBottom: '0.5rem', fontSize: '1rem' }}>Adversarial Robustness (Rank-1 Accuracy Drop-off)</h4>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
              Comparing model accuracy under various synthetic adversarial degradations. A smaller drop from "Clean" indicates higher robustness.
            </p>
            <div style={{ width: '100%', height: 250 }}>
              <ResponsiveContainer>
                <BarChart data={robustnessData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.2} vertical={false}/>
                  <XAxis dataKey="name" stroke="var(--text-secondary)" fontSize={12} />
                  <YAxis domain={[0, 100]} stroke="var(--text-secondary)" fontSize={12} unit="%" />
                  <RechartsTooltip cursor={{fill: 'rgba(255,255,255,0.05)'}} contentStyle={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-light)', borderRadius: '8px' }} />
                  <Bar dataKey="value" fill="var(--danger)" radius={[4, 4, 0, 0]} maxBarSize={80} name="Rank-1 Accuracy" label={{ position: 'top', fill: 'var(--text-secondary)', fontSize: 12, formatter: (val) => `${Number(val).toFixed(2)}%` }} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

        </div>
      )}
    </div>
  );
}

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [stats, setStats] = useState({ videos_processed: 0, total_detections: 0, unique_identities: 0, reid_matches: 0 });

  useEffect(() => {
    // Fetch stats on load
    fetch('/api/stats')
      .then(res => res.json())
      .then(data => setStats(data))
      .catch(err => console.error(err));
  }, [activeTab]); // Refresh stats when tab changes

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <DashboardView stats={stats} />;
      case 'ranking':
      case 'search':
        return <ImageRankingView />;
      case 'upload':
        return <UploadView />;
      case 'identities':
        return <IdentitiesView />;
      case 'analytics':
        return <AnalyticsView />;
      case 'settings':
        return <SettingsView />;
      case 'datasets':
        return <DatasetsView />;
      default:
        return (
          <div className="panel" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '300px', color: 'var(--text-secondary)' }}>
            <h2>Module "{activeTab}" is coming soon!</h2>
          </div>
        );
    }
  };

  return (
    <div className={`app-layout ${isDarkMode ? 'dark' : ''}`}>
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-logo"><UserCheck size={20} /></div>
          <div className="sidebar-title">
            <h2>ReID Vision</h2>
            <p>Person Detection & Re-ID</p>
          </div>
        </div>
        
        <nav className="sidebar-nav">
          <button className={`nav-item ${activeTab === 'dashboard' ? 'active' : ''}`} onClick={() => setActiveTab('dashboard')}>
            <LayoutDashboard size={18} /> Dashboard
          </button>
          <button className={`nav-item ${activeTab === 'upload' ? 'active' : ''}`} onClick={() => setActiveTab('upload')}>
            <Upload size={18} /> Upload Video / Image
          </button>
          <button className={`nav-item ${activeTab === 'camera' ? 'active' : ''}`} onClick={() => setActiveTab('camera')}>
            <Camera size={18} /> Real-time Camera
          </button>
          <button className={`nav-item ${activeTab === 'search' ? 'active' : ''}`} onClick={() => setActiveTab('search')}>
            <Search size={18} /> Image Ranking
          </button>
          <button className={`nav-item ${activeTab === 'identities' ? 'active' : ''}`} onClick={() => setActiveTab('identities')}>
            <Users size={18} /> Tracks & Identities
          </button>
          <button className={`nav-item ${activeTab === 'analytics' ? 'active' : ''}`} onClick={() => setActiveTab('analytics')}>
            <BarChart3 size={18} /> Analytics
          </button>
          <button className={`nav-item ${activeTab === 'datasets' ? 'active' : ''}`} onClick={() => setActiveTab('datasets')}>
            <Database size={18} /> Datasets
          </button>
          <div style={{ flex: 1 }}></div>
          <button className={`nav-item ${activeTab === 'settings' ? 'active' : ''}`} onClick={() => setActiveTab('settings')}>
            <Settings size={18} /> Settings
          </button>
        </nav>
        
        {/* Sidebar Footer */}
        <div className="sidebar-footer">
          <button className="nav-item" onClick={() => setIsDarkMode(!isDarkMode)}>
            {isDarkMode ? <Sun size={20} /> : <Moon size={20} />}
            <span>{isDarkMode ? 'Light' : 'Dark'}</span>
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <header className="topbar">
          <div>
            <h1 className="topbar-title" style={{textTransform: 'capitalize'}}>
              {activeTab.replace('-', ' ')}
            </h1>
            <p className="topbar-subtitle">Monitor detection and re-identification results, track identities and analyze insights.</p>
          </div>
          <div className="topbar-actions">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', padding: '0.4rem 0.75rem', background: 'rgba(16, 185, 129, 0.1)', color: 'var(--success)', borderRadius: '999px', fontSize: '0.75rem', fontWeight: 600 }}>
              <div style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'var(--success)' }}></div>
              System Online
            </div>
          </div>
        </header>

        <div className="content-scroll">
          {renderContent()}
        </div>
      </main>
    </div>
  );
}

export default App;
