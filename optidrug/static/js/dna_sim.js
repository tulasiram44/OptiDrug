/**
 * OptiDrug - Interactive 3D DNA Double Helix & Biological Propagation Simulation
 * 
 * Features:
 * - Real-time 3D projected Canvas rendering with depth-sorting, phong-like shading, and glow effects
 * - Full 3D rotation, pitch, and orbit via mouse drag and touch
 * - Nucleotide base pairs: Adenine (Red), Thymine (Cyan), Guanine (Green), Cytosine (Gold)
 * - Dynamic Oncogenic Mutation Trigger with particle burst and base substitution animation
 * - Multi-scale biological effect propagation: DNA -> Codon -> Protein Kinase Pocket -> Candidate Drug Rescue
 */

class DNASimulator {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!canvasId || !this.canvas) return;
        this.ctx = this.canvas.getContext('2d');
        
        // Dimensions & State
        this.width = this.canvas.width;
        this.height = this.canvas.height;
        this.isMutant = false;
        this.isDrugDocked = false;
        this.currentCandidate = null;
        this.isPaused = false;
        this.speed = 0.015;
        this.angle = 0;
        this.pitch = 0.25;
        this.roll = 0.0;
        
        // Interactive Mouse Orbit
        this.isDragging = false;
        this.lastMouseX = 0;
        this.lastMouseY = 0;
        this.zoom = 1.0;
        
        // Simulation Data
        this.targetData = null;
        this.mutationData = null;
        this.dnaSequence = "ATGCGATTCCTGAGCGTACTG";
        this.mutantSequence = "ATGCGATTCCGGAGCGTACTG";
        this.mutationIndex = 10;
        
        // Particle system for mutation transition & drug docking
        this.particles = [];
        this.pulseTime = 0;
        
        this.initEventListeners();
        this.resize();
        this.animate();
    }
    
    initEventListeners() {
        window.addEventListener('resize', () => this.resize());
        
        this.canvas.addEventListener('mousedown', (e) => {
            this.isDragging = true;
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
        });
        
        window.addEventListener('mousemove', (e) => {
            if (!this.isDragging) return;
            const deltaX = e.clientX - this.lastMouseX;
            const deltaY = e.clientY - this.lastMouseY;
            this.angle += deltaX * 0.008;
            this.pitch += deltaY * 0.008;
            this.pitch = Math.max(-1.2, Math.min(1.2, this.pitch));
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
        });
        
        window.addEventListener('mouseup', () => {
            this.isDragging = false;
        });
        
        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            this.zoom += e.deltaY * -0.001;
            this.zoom = Math.max(0.6, Math.min(1.8, this.zoom));
        }, { passive: false });
    }
    
    resize() {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.width = rect.width;
        this.height = Math.max(420, rect.height || 420);
        this.canvas.width = this.width;
        this.canvas.height = this.height;
    }
    
    loadTarget(targetInfo) {
        this.targetData = targetInfo;
        if (targetInfo && targetInfo.dna_simulation) {
            this.mutationData = targetInfo.dna_simulation;
            this.dnaSequence = this.mutationData.wildtype_dna;
            this.mutantSequence = this.mutationData.mutant_dna;
            this.mutationIndex = this.mutationData.mutation_index;
        }
        this.isMutant = false;
        this.isDrugDocked = false;
        this.triggerParticles(0, 0, '#3b82f6', 20);
        this.updateDOM();
    }
    
    toggleMutation(forceMutant = null) {
        const prev = this.isMutant;
        this.isMutant = (forceMutant !== null) ? forceMutant : !this.isMutant;
        if (prev !== this.isMutant) {
            this.triggerParticles(this.width / 2, this.height / 2, this.isMutant ? '#ef4444' : '#10b981', 40);
            this.updateDOM();
        }
    }
    
    dockDrug(candidate) {
        this.currentCandidate = candidate;
        this.isDrugDocked = true;
        this.triggerParticles(this.width / 2, this.height / 2, '#8b5cf6', 50);
        this.updateDOM();
    }
    
    undockDrug() {
        this.isDrugDocked = false;
        this.updateDOM();
    }
    
    triggerParticles(x, y, color, count = 30) {
        for (let i = 0; i < count; i++) {
            const angle = Math.random() * Math.PI * 2;
            const speed = 1.5 + Math.random() * 4.5;
            this.particles.push({
                x: x || this.width / 2,
                y: y || this.height / 2,
                vx: Math.cos(angle) * speed,
                vy: Math.sin(angle) * speed,
                life: 1.0,
                decay: 0.02 + Math.random() * 0.03,
                color: color,
                size: 2.5 + Math.random() * 3.5
            });
        }
    }
    
    getComplement(base) {
        switch (base) {
            case 'A': return 'T';
            case 'T': return 'A';
            case 'G': return 'C';
            case 'C': return 'G';
            default: return 'A';
        }
    }
    
    getBaseColor(base) {
        switch (base) {
            case 'A': return '#ef4444'; // Red
            case 'T': return '#06b6d4'; // Cyan
            case 'G': return '#10b981'; // Green
            case 'C': return '#f59e0b'; // Gold
            default: return '#94a3b8';
        }
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        if (!this.isPaused && !this.isDragging) {
            this.angle += this.speed;
        }
        this.pulseTime += 0.05;
        this.render();
    }
    
    render() {
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.width, this.height);
        
        // Deep modern bio-tech background
        const grad = ctx.createLinearGradient(0, 0, this.width, this.height);
        grad.addColorStop(0, '#0f172a');
        grad.addColorStop(0.5, '#1e293b');
        grad.addColorStop(1, '#090d16');
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, this.width, this.height);
        
        // Render subtle 3D coordinate grid / field
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
        ctx.lineWidth = 1;
        for (let x = 0; x < this.width; x += 40) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, this.height);
            ctx.stroke();
        }
        for (let y = 0; y < this.height; y += 40) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(this.width, y);
            ctx.stroke();
        }
        
        // Active DNA sequence based on mutation state
        const currentSeq = this.isMutant ? this.mutantSequence : this.dnaSequence;
        const numPairs = currentSeq.length;
        
        const centerX = this.width / 2;
        const centerY = this.height / 2;
        const radius = 80 * this.zoom;
        const verticalSpacing = 19 * this.zoom;
        const totalHeight = numPairs * verticalSpacing;
        const startY = centerY - totalHeight / 2;
        
        // Depth-sorting array for 3D elements
        const renderQueue = [];
        
        for (let i = 0; i < numPairs; i++) {
            const base1 = currentSeq[i];
            const base2 = this.getComplement(base1);
            const isTargetCodon = (i >= this.mutationIndex - 1 && i <= this.mutationIndex + 1);
            const isExactMutatedBase = (i === this.mutationIndex);
            
            const theta = this.angle + (i * 0.42);
            const y = startY + (i * verticalSpacing);
            
            // 3D rotation coordinates with pitch
            const cosT = Math.cos(theta);
            const sinT = Math.sin(theta);
            const cosP = Math.cos(this.pitch);
            const sinP = Math.sin(this.pitch);
            
            // Strand 1 (5' -> 3')
            const x1_raw = radius * cosT;
            const z1_raw = radius * sinT;
            const x1 = centerX + x1_raw;
            const y1 = y * cosP - z1_raw * sinP;
            const z1 = y * sinP + z1_raw * cosP;
            
            // Strand 2 (3' -> 5', 180 degrees opposite)
            const x2_raw = -x1_raw;
            const z2_raw = -z1_raw;
            const x2 = centerX + x2_raw;
            const y2 = y * cosP - z2_raw * sinP;
            const z2 = y * sinP + z2_raw * cosP;
            
            // Base pair rung center
            const zMid = (z1 + z2) / 2;
            
            renderQueue.push({
                type: 'rung',
                i,
                base1,
                base2,
                x1, y1, z1,
                x2, y2, z2,
                z: zMid,
                isTargetCodon,
                isExactMutatedBase
            });
            
            renderQueue.push({
                type: 'backbone_node',
                strand: 1,
                x: x1, y: y1, z: z1,
                isExactMutatedBase
            });
            
            renderQueue.push({
                type: 'backbone_node',
                strand: 2,
                x: x2, y: y2, z: z2,
                isExactMutatedBase
            });
        }
        
        // Sort from back to front (Painter's algorithm)
        renderQueue.sort((a, b) => a.z - b.z);
        
        // Render 3D elements
        for (const item of renderQueue) {
            if (item.type === 'rung') {
                this.renderRung(ctx, item);
            } else if (item.type === 'backbone_node') {
                this.renderBackboneNode(ctx, item);
            }
        }
        
        // Render Active Docked Drug Molecule overlay
        if (this.isDrugDocked) {
            this.renderDockedDrug(ctx, centerX, centerY);
        }
        
        // Render particles
        this.renderParticles(ctx);
        
        // Render HUD Annotations & Overlay
        this.renderHUD(ctx);
    }
    
    renderRung(ctx, item) {
        const { x1, y1, x2, y2, z, base1, base2, isExactMutatedBase } = item;
        const color1 = this.getBaseColor(base1);
        const color2 = this.getBaseColor(base2);
        
        // Depth-based alpha
        const depthAlpha = Math.max(0.25, Math.min(1.0, 0.6 + (z / 250)));
        
        // Rung hydrogen bond connector
        const midX = (x1 + x2) / 2;
        const midY = (y1 + y2) / 2;
        
        ctx.save();
        ctx.globalAlpha = depthAlpha;
        
        // Base 1 segment
        ctx.strokeStyle = color1;
        ctx.lineWidth = isExactMutatedBase ? 4 : 2.5;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(midX, midY);
        ctx.stroke();
        
        // Base 2 segment
        ctx.strokeStyle = color2;
        ctx.lineWidth = isExactMutatedBase ? 4 : 2.5;
        ctx.beginPath();
        ctx.moveTo(midX, midY);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        
        // Hydrogen bonds in middle (dots)
        ctx.fillStyle = '#ffffff';
        const numHbonds = (base1 === 'G' || base1 === 'C') ? 3 : 2;
        for (let h = -numHbonds / 2; h <= numHbonds / 2; h++) {
            ctx.beginPath();
            ctx.arc(midX + h * 3, midY, 1.2, 0, Math.PI * 2);
            ctx.fill();
        }
        
        // If this is the mutated codon base
        if (isExactMutatedBase) {
            const pulse = 1.0 + 0.35 * Math.sin(this.pulseTime * 4);
            ctx.fillStyle = this.isMutant ? '#ef4444' : '#10b981';
            ctx.beginPath();
            ctx.arc(midX, midY, 8 * pulse, 0, Math.PI * 2);
            ctx.fill();
            
            // Warning Beacon Ring
            ctx.strokeStyle = this.isMutant ? 'rgba(239, 68, 68, 0.7)' : 'rgba(16, 185, 129, 0.7)';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(midX, midY, 16 * pulse, 0, Math.PI * 2);
            ctx.stroke();
            
            // Label
            ctx.font = 'bold 12px monospace';
            ctx.fillStyle = '#ffffff';
            ctx.textAlign = 'center';
            const label = this.isMutant ? `MUTANT [${base1}]` : `WT [${base1}]`;
            ctx.fillText(label, midX, midY - 22);
        }
        
        ctx.restore();
    }
    
    renderBackboneNode(ctx, item) {
        const { x, y, z, strand, isExactMutatedBase } = item;
        const depthAlpha = Math.max(0.3, Math.min(1.0, 0.65 + (z / 250)));
        const nodeRadius = (strand === 1 ? 4.5 : 4.0) * (0.8 + z / 300);
        
        ctx.save();
        ctx.globalAlpha = depthAlpha;
        
        const glowColor = isExactMutatedBase 
            ? (this.isMutant ? '#ef4444' : '#10b981')
            : (strand === 1 ? '#3b82f6' : '#a855f7');
            
        ctx.fillStyle = glowColor;
        ctx.shadowColor = glowColor;
        ctx.shadowBlur = isExactMutatedBase ? 15 : 6;
        
        ctx.beginPath();
        ctx.arc(x, y, Math.max(2, nodeRadius), 0, Math.PI * 2);
        ctx.fill();
        
        ctx.restore();
    }
    
    renderDockedDrug(ctx, cx, cy) {
        const pulse = 1.0 + 0.15 * Math.sin(this.pulseTime * 3);
        const name = this.currentCandidate ? this.currentCandidate.compound_id : "OPT-CANDIDATE";
        
        ctx.save();
        // Molecular docking boundary box
        ctx.strokeStyle = '#a855f7';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.strokeRect(cx - 110, cy - 60, 220, 120);
        ctx.setLineDash([]);
        
        // Small Molecule Ligand Ring Core
        ctx.fillStyle = 'rgba(168, 85, 247, 0.25)';
        ctx.beginPath();
        ctx.arc(cx, cy, 38 * pulse, 0, Math.PI * 2);
        ctx.fill();
        
        // Hexagonal scaffold skeleton
        ctx.strokeStyle = '#c084fc';
        ctx.lineWidth = 3;
        ctx.beginPath();
        for (let i = 0; i < 6; i++) {
            const rad = 26 * pulse;
            const a = (i * Math.PI) / 3;
            const px = cx + rad * Math.cos(a);
            const py = cy + rad * Math.sin(a);
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
        }
        ctx.closePath();
        ctx.stroke();
        
        // H-bond dotted arrows docking into kinase hinge (Met793)
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(cx - 25, cy);
        ctx.lineTo(cx - 85, cy);
        ctx.stroke();
        ctx.setLineDash([]);
        
        ctx.fillStyle = '#38bdf8';
        ctx.font = 'bold 11px sans-serif';
        ctx.fillText("H-bond (Met793)", cx - 85, cy - 8);
        
        // Drug Label
        ctx.fillStyle = '#f8fafc';
        ctx.font = 'bold 13px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`LIGAND DOCKED: ${name}`, cx, cy + 50);
        ctx.fillStyle = '#4ade80';
        ctx.font = '11px sans-serif';
        ctx.fillText("✓ Kinase Hyperactivity Inhibited", cx, cy + 68);
        
        ctx.restore();
    }
    
    renderParticles(ctx) {
        for (let i = this.particles.length - 1; i >= 0; i--) {
            const p = this.particles[i];
            p.x += p.vx;
            p.y += p.vy;
            p.life -= p.decay;
            
            if (p.life <= 0) {
                this.particles.splice(i, 1);
                continue;
            }
            
            ctx.save();
            ctx.globalAlpha = p.life;
            ctx.fillStyle = p.color;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
        }
    }
    
    renderHUD(ctx) {
        ctx.save();
        ctx.font = '12px system-ui, sans-serif';
        ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
        ctx.fillText("3D Orbit: Drag mouse | Zoom: Scroll", 14, 24);
        
        // Legend for base pairs
        const legendX = this.width - 160;
        const legendY = 24;
        
        const bases = [
            { name: 'A (Adenine)', color: '#ef4444' },
            { name: 'T (Thymine)', color: '#06b6d4' },
            { name: 'G (Guanine)', color: '#10b981' },
            { name: 'C (Cytosine)', color: '#f59e0b' }
        ];
        
        bases.forEach((b, idx) => {
            ctx.fillStyle = b.color;
            ctx.beginPath();
            ctx.arc(legendX, legendY + idx * 16, 4, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = 'rgba(255, 255, 255, 0.75)';
            ctx.font = '10px monospace';
            ctx.fillText(b.name, legendX + 12, legendY + idx * 16 + 3);
        });
        
        ctx.restore();
    }
    
    updateDOM() {
        const sim = this.mutationData;
        if (!sim) return;
        
        // Update status badge
        const badge = document.getElementById('dna-status-badge');
        if (badge) {
            if (this.isMutant) {
                badge.className = 'status-pill mutant';
                badge.innerHTML = `🔴 ONCOGENIC MUTANT: ${sim.mutation_name}`;
            } else {
                badge.className = 'status-pill wildtype';
                badge.innerHTML = `🟢 WILD-TYPE (Healthy): ${sim.wt_codon}`;
            }
        }
        
        // Update Codon info
        const codonEl = document.getElementById('sim-codon-impact');
        if (codonEl) {
            codonEl.innerHTML = this.isMutant ? `
                <div class="impact-alert mutant">
                    <b>Genomic DNA Variant:</b> <code>${sim.mutation_name}</code> at <b>${sim.gene_locus}</b><br>
                    <b>Codon Alteration:</b> <code>${sim.wt_codon}</code> ➔ <code>${sim.mut_codon}</code><br>
                    <b>Chemical Shift:</b> ${sim.chemical_change}
                </div>
            ` : `
                <div class="impact-alert wildtype">
                    <b>Physiological Wild-Type:</b> <code>${sim.gene_locus}</code><br>
                    <b>Normal Codon:</b> <code>${sim.wt_codon}</code><br>
                    <b>Normal Function:</b> Regulated basal enzymatic receptor signaling.
                </div>
            `;
        }
        
        // Update Protein Conformation impact
        const proteinEl = document.getElementById('sim-protein-impact');
        if (proteinEl) {
            proteinEl.innerHTML = this.isMutant ? `
                <div class="consequence-box mutant">
                    <h5>🚨 Protein Structural Hyperactivation</h5>
                    <p>${sim.structural_impact}</p>
                    <div class="cellular-effect"><b>Downstream Pathology:</b> ${sim.cellular_consequence}</div>
                </div>
            ` : `
                <div class="consequence-box wildtype">
                    <h5>✓ Normal Kinase Autoinhibition</h5>
                    <p>Receptor kinase maintains normal closed regulatory conformation, strictly dependent on physiological ligand activation.</p>
                </div>
            `;
        }
        
        // Update Rescue info
        const rescueEl = document.getElementById('sim-rescue-impact');
        if (rescueEl) {
            if (this.isDrugDocked && this.currentCandidate) {
                rescueEl.innerHTML = `
                    <div class="docking-box active">
                        <h5>🛡️ Small Molecule Docking Success: ${this.currentCandidate.compound_id}</h5>
                        <p><b>Scaffold:</b> ${this.currentCandidate.scaffold_type} | <b>Pred. pIC50:</b> ${this.currentCandidate.predicted_pic50.toFixed(2)} | <b>Composite Score:</b> ${this.currentCandidate.composite_score.toFixed(1)}/100</p>
                        <p style="font-size:0.85rem; color:#cbd5e1;">${sim.drug_rescue}</p>
                    </div>
                `;
            } else {
                rescueEl.innerHTML = `
                    <div class="docking-box inactive">
                        <h5>Select a candidate compound to simulate active-site rescue</h5>
                        <p style="font-size:0.85rem; color:#94a3b8;">Click "Simulate Docking" on any candidate from the prioritized lead list below to visualize how the compound locks into the mutated ATP cleft.</p>
                    </div>
                `;
            }
        }
    }
}

// Global instance
window.DNASimulator = DNASimulator;
