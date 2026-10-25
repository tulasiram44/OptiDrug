/**
 * OptiDrug - Main Dashboard JavaScript Application
 * Coordinates REST API calls, Plotly data visualizations, candidate tables,
 * multi-objective slider reactivity, and triggers 3D DNA simulation.
 */

let dnaSim = null;
let currentPipelineData = null;
let currentTargetId = "EGFR";

document.addEventListener("DOMContentLoaded", async () => {
    // 1. Initialize 3D DNA Simulation
    dnaSim = new DNASimulator("dna-canvas");

    // 2. Setup Tabs
    const tabBtns = document.querySelectorAll(".tab-btn");
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".tab-content").forEach(tc => tc.classList.remove("active"));
            btn.classList.add("active");
            const targetTab = document.getElementById(btn.dataset.tab);
            if (targetTab) targetTab.classList.add("active");
            
            // Trigger Plotly relayout on tab switch if needed
            window.dispatchEvent(new Event('resize'));
        });
    });

    // 3. Load Targets and run initial pipeline
    await loadTargetsList();
    await runPipeline();

    // 4. Setup Event Listeners
    setupControls();
});

async function loadTargetsList() {
    try {
        const res = await fetch("/api/targets");
        const data = await res.json();
        if (data.status === "success") {
            const select = document.getElementById("target-select");
            select.innerHTML = "";
            data.targets.forEach(t => {
                const opt = document.createElement("option");
                opt.value = t.target_id;
                opt.textContent = `${t.target_id} — ${t.disease}`;
                select.appendChild(opt);
            });
            select.value = currentTargetId;
            select.addEventListener("change", (e) => {
                currentTargetId = e.target.value;
                runPipeline();
            });
        }
    } catch (e) {
        console.error("Failed to load targets", e);
    }
}

function setupControls() {
    // Re-run Pipeline Button
    document.getElementById("btn-run-pipeline")?.addEventListener("click", () => {
        runPipeline();
    });

    // DNA Mutation Toggle Button
    document.getElementById("btn-toggle-mutant")?.addEventListener("click", () => {
        if (!dnaSim) return;
        dnaSim.toggleMutation();
        const btn = document.getElementById("btn-toggle-mutant");
        if (dnaSim.isMutant) {
            btn.classList.add("active-mutant");
            btn.textContent = "Switch to Wild-Type (Healthy)";
        } else {
            btn.classList.remove("active-mutant");
            btn.textContent = "Simulate Oncogenic Mutation";
        }
    });

    // DNA Pause/Resume Button
    document.getElementById("btn-pause-dna")?.addEventListener("click", () => {
        if (!dnaSim) return;
        dnaSim.isPaused = !dnaSim.isPaused;
        const btn = document.getElementById("btn-pause-dna");
        btn.textContent = dnaSim.isPaused ? "▶ Resume Spin" : "⏸ Pause Spin";
    });

    // Weight sliders reactivity
    const weightSliders = ["w_eff", "w_safe", "w_inter", "w_cost", "w_drug"];
    weightSliders.forEach(id => {
        const slider = document.getElementById(id);
        const valSpan = document.getElementById(`${id}-val`);
        if (slider && valSpan) {
            slider.addEventListener("input", () => {
                valSpan.textContent = `${(parseFloat(slider.value) * 100).toFixed(0)}%`;
                debounceReweight();
            });
        }
    });

    // Search and filters
    document.getElementById("candidate-search")?.addEventListener("input", filterAndRenderTable);
    document.getElementById("filter-pareto")?.addEventListener("change", filterAndRenderTable);
    document.getElementById("filter-lipinski")?.addEventListener("change", filterAndRenderTable);
    document.getElementById("filter-scaffold")?.addEventListener("change", filterAndRenderTable);

    // CSV Download
    document.getElementById("btn-download-csv")?.addEventListener("click", downloadCSV);
}

let reweightTimeout = null;
function debounceReweight() {
    clearTimeout(reweightTimeout);
    reweightTimeout = setTimeout(executeReweight, 250);
}

async function runPipeline() {
    showLoading(true);
    const targetId = document.getElementById("target-select").value || currentTargetId;
    const nSamples = document.getElementById("input-n-samples")?.value || 250;
    const seed = document.getElementById("input-seed")?.value || 42;

    const w_eff = parseFloat(document.getElementById("w_eff")?.value || 0.35);
    const w_safe = parseFloat(document.getElementById("w_safe")?.value || 0.25);
    const w_inter = parseFloat(document.getElementById("w_inter")?.value || 0.15);
    const w_cost = parseFloat(document.getElementById("w_cost")?.value || 0.15);
    const w_drug = parseFloat(document.getElementById("w_drug")?.value || 0.10);

    try {
        const res = await fetch("/api/pipeline/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                target_id: targetId,
                n_samples: nSamples,
                seed: seed,
                w_eff, w_safe, w_inter, w_cost, w_drug
            })
        });
        const data = await res.json();
        if (data.status === "success") {
            currentPipelineData = data;
            
            // 1. Load target into DNA Simulation
            if (dnaSim && data.target) {
                dnaSim.loadTarget(data.target);
            }

            // 2. Update Metrics & Audit logs
            updateMetrics(data);

            // 3. Render Visualizations
            renderEvaluationPlot(data.evaluation_points, data.metrics);
            renderFeatureImportancePlot(data.feature_importances);
            renderParetoPlot(data.candidates);
            renderRadarPlot(data.candidates.slice(0, 4));

            // 4. Update Target Info & NCBI Details
            await updateTargetBiology(targetId);

            // 5. Populate Candidate Table & Scaffold Filters
            populateScaffoldFilter(data.candidates);
            filterAndRenderTable();

            // 6. Select #1 lead for inspection
            if (data.candidates.length > 0) {
                inspectCandidate(data.candidates[0]);
            }
        }
    } catch (e) {
        console.error("Pipeline run failed", e);
    } finally {
        showLoading(false);
    }
}

async function executeReweight() {
    if (!currentPipelineData) return;
    const targetId = currentPipelineData.target.target_id;
    const w_eff = parseFloat(document.getElementById("w_eff")?.value || 0.35);
    const w_safe = parseFloat(document.getElementById("w_safe")?.value || 0.25);
    const w_inter = parseFloat(document.getElementById("w_inter")?.value || 0.15);
    const w_cost = parseFloat(document.getElementById("w_cost")?.value || 0.15);
    const w_drug = parseFloat(document.getElementById("w_drug")?.value || 0.10);

    try {
        const res = await fetch("/api/pipeline/reweight", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                target_id: targetId,
                w_eff, w_safe, w_inter, w_cost, w_drug
            })
        });
        const data = await res.json();
        if (data.status === "success") {
            const reweightMap = new Map();
            data.candidates.forEach(c => reweightMap.set(c.compound_id, c));

            // Update candidate objects in current data
            currentPipelineData.candidates.forEach(c => {
                const updated = reweightMap.get(c.compound_id);
                if (updated) {
                    c.optimized_rank = updated.optimized_rank;
                    c.preliminary_rank = updated.preliminary_rank;
                    c.rank_delta = updated.rank_delta;
                    c.composite_score = updated.composite_score;
                    c.desirability_index = updated.desirability_index;
                    c.is_pareto_optimal = updated.is_pareto_optimal;
                }
            });

            // Re-sort by optimized rank
            currentPipelineData.candidates.sort((a, b) => a.optimized_rank - b.optimized_rank);

            // Re-render plots and table
            renderParetoPlot(currentPipelineData.candidates);
            renderRadarPlot(currentPipelineData.candidates.slice(0, 4));
            filterAndRenderTable();
        }
    } catch (e) {
        console.error("Reweight failed", e);
    }
}

async function updateTargetBiology(targetId) {
    try {
        const res = await fetch(`/api/target/${targetId}`);
        const data = await res.json();
        if (data.status === "success") {
            const t = data.target;
            const ncbi = data.ncbi_gene;
            
            const bioCard = document.getElementById("target-bio-details");
            if (bioCard) {
                bioCard.innerHTML = `
                    <div style="margin-bottom:12px;">
                        <h4 style="color:#60a5fa; margin-bottom:4px;">${t.disease}</h4>
                        <div style="font-size:0.9rem; color:#94a3b8;"><b>Target Gene:</b> <code>${t.gene_symbol}</code> | <b>UniProt:</b> <a href="https://www.uniprot.org/uniprotkb/${t.uniprot_id}" target="_blank" style="color:#38bdf8;">${t.uniprot_id}</a> | <b>NCBI:</b> <a href="https://www.ncbi.nlm.nih.gov/gene/${t.ncbi_gene_id}" target="_blank" style="color:#38bdf8;">${t.ncbi_gene_id}</a></div>
                    </div>
                    <div style="background:#172138; padding:12px 16px; border-radius:8px; margin-bottom:12px; font-size:0.88rem;">
                        <p style="margin-bottom:6px;"><b>Target Protein:</b> ${t.protein_name}</p>
                        <p style="margin-bottom:6px;"><b>Binding Pocket:</b> ${t.drug_target_pocket}</p>
                        <p style="margin-bottom:6px;"><b>Biological Mechanism:</b> ${t.biological_mechanism}</p>
                        <p><b>Approved Benchmark Drugs:</b> ${t.benchmark_drugs.map(d => `<span class="badge" style="background:rgba(59,130,246,0.15); color:#60a5fa; margin-right:4px;">${d}</span>`).join('')}</p>
                    </div>
                    <div style="background:rgba(16,185,129,0.08); border-left:3px solid #10b981; padding:10px 14px; border-radius:4px; font-size:0.85rem;">
                        <b>NCBI Entrez Record:</b> ${ncbi.summary}
                        <div style="margin-top:4px; color:#64748b; font-size:0.78rem;">Source: ${ncbi.source} | Aliases: ${ncbi.aliases || 'N/A'}</div>
                    </div>
                `;
            }

            // PubMed Articles
            const pubmedEl = document.getElementById("target-pubmed-list");
            if (pubmedEl && data.pubmed_articles) {
                pubmedEl.innerHTML = data.pubmed_articles.map(art => `
                    <div style="background:#172138; border-left:3px solid #3b82f6; padding:8px 12px; border-radius:6px; margin-bottom:8px; font-size:0.84rem;">
                        <a href="${art.url}" target="_blank" style="color:#93c5fd; font-weight:600; text-decoration:none;">${art.title}</a>
                        <div style="color:#64748b; font-size:0.78rem; margin-top:2px;">${art.authors} — <i>${art.source}</i> (${art.pubdate}) [PMID: ${art.pmid}]</div>
                    </div>
                `).join('');
            }
        }
    } catch (e) {
        console.error("Failed to load biology", e);
    }
}

function updateMetrics(data) {
    const m = data.metrics;
    const a = data.audit_log;

    document.getElementById("metric-r2").textContent = m.test_r2.toFixed(4);
    document.getElementById("metric-mae").textContent = m.test_mae.toFixed(4);
    document.getElementById("metric-rmse").textContent = m.test_rmse.toFixed(4);
    document.getElementById("metric-pearson").textContent = m.test_pearson.toFixed(4);

    document.getElementById("audit-initial").textContent = a.initial_rows;
    document.getElementById("audit-dedup").textContent = a.duplicate_rows_removed;
    const missingTotal = Object.values(a.missing_values_imputed).reduce((x, y) => x + y, 0);
    document.getElementById("audit-imputed").textContent = missingTotal;
    document.getElementById("audit-train").textContent = `${a.train_rows} (80%)`;
    document.getElementById("audit-test").textContent = `${a.test_rows} (20%)`;
}

function renderEvaluationPlot(evalPoints, metrics) {
    const yTrue = evalPoints.y_test;
    const yPred = evalPoints.y_test_pred;
    const minVal = Math.min(...yTrue, ...yPred) - 0.05;
    const maxVal = Math.max(...yTrue, ...yPred) + 0.05;

    const trace1 = {
        x: yTrue,
        y: yPred,
        mode: 'markers',
        type: 'scatter',
        name: 'Held-out Test Compounds',
        marker: { color: '#38bdf8', size: 8, opacity: 0.8 }
    };

    const trace2 = {
        x: [minVal, maxVal],
        y: [minVal, maxVal],
        mode: 'lines',
        type: 'scatter',
        name: 'Identity (y = x)',
        line: { color: '#ef4444', dash: 'dash', width: 2 }
    };

    const layout = {
        title: { text: `<b>Random Forest QSAR Evaluation (R² = ${metrics.test_r2.toFixed(3)}, MAE = ${metrics.test_mae.toFixed(3)})</b>`, font: { color: '#f8fafc', size: 14 } },
        xaxis: { title: 'Actual Activity Score', color: '#94a3b8', gridcolor: '#1e293b' },
        yaxis: { title: 'Predicted Activity Score', color: '#94a3b8', gridcolor: '#1e293b' },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        margin: { l: 45, r: 25, t: 45, b: 40 },
        height: 380,
        legend: { font: { color: '#cbd5e1' }, orientation: 'h', y: -0.2 }
    };

    Plotly.newPlot('eval-plot', [trace1, trace2], layout, { responsive: true, displayModeBar: false });
}

function renderFeatureImportancePlot(fiList) {
    const sorted = [...fiList].sort((a, b) => a.importance - b.importance);
    const trace = {
        x: sorted.map(d => d.importance_pct),
        y: sorted.map(d => d.feature.replace(/_/g, ' ').toUpperCase()),
        type: 'bar',
        orientation: 'h',
        marker: {
            color: sorted.map(d => d.importance_pct),
            colorscale: 'Viridis'
        },
        text: sorted.map(d => `${d.importance_pct.toFixed(1)}%`),
        textposition: 'outside'
    };

    const layout = {
        title: { text: '<b>Feature Importance (% Variance Explained)</b>', font: { color: '#f8fafc', size: 14 } },
        xaxis: { title: 'Relative Importance (%)', color: '#94a3b8', gridcolor: '#1e293b' },
        yaxis: { color: '#94a3b8', gridcolor: '#1e293b' },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        margin: { l: 140, r: 40, t: 45, b: 40 },
        height: 380
    };

    Plotly.newPlot('feature-plot', [trace], layout, { responsive: true, displayModeBar: false });
}

function renderParetoPlot(candidates) {
    const nonPareto = candidates.filter(c => !c.is_pareto_optimal);
    const pareto = candidates.filter(c => c.is_pareto_optimal);

    const trace1 = {
        x: nonPareto.map(c => 1.0 - c.safety_risk),
        y: nonPareto.map(c => c.predicted_pic50),
        text: nonPareto.map(c => `${c.compound_id} (${c.compound_name})<br>Composite: ${c.composite_score.toFixed(1)}`),
        mode: 'markers',
        type: 'scatter',
        name: 'Candidates',
        marker: {
            color: nonPareto.map(c => c.composite_score),
            colorscale: 'Blues',
            size: 7,
            opacity: 0.65
        }
    };

    const trace2 = {
        x: pareto.map(c => 1.0 - c.safety_risk),
        y: pareto.map(c => c.predicted_pic50),
        text: pareto.map(c => `⭐ ${c.compound_id} (${c.compound_name})<br>Rank: #${c.optimized_rank}<br>Composite: ${c.composite_score.toFixed(1)}`),
        mode: 'markers',
        type: 'scatter',
        name: 'Pareto Optimal Frontier ⭐',
        marker: {
            symbol: 'star',
            color: '#f59e0b',
            size: 14,
            line: { color: '#78350f', width: 1.5 }
        }
    };

    const layout = {
        title: { text: '<b>Multi-Objective Pareto Trade-Off (Potency vs Safety)</b>', font: { color: '#f8fafc', size: 14 } },
        xaxis: { title: 'Safety Profile (1 - Risk, Higher is Safer)', color: '#94a3b8', gridcolor: '#1e293b' },
        yaxis: { title: 'Predicted Bioactivity (pIC50, Higher is Potent)', color: '#94a3b8', gridcolor: '#1e293b' },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        margin: { l: 45, r: 25, t: 45, b: 40 },
        height: 400,
        legend: { font: { color: '#cbd5e1' }, orientation: 'h', y: -0.2 }
    };

    Plotly.newPlot('pareto-plot', [trace1, trace2], layout, { responsive: true, displayModeBar: false });
}

function renderRadarPlot(topCandidates) {
    const categories = ['Effectiveness', 'Safety Profile', 'Low Interaction', 'Synthetic Economy', 'Drug-likeness'];
    const colors = ['#38bdf8', '#34d399', '#fbbf24', '#c084fc'];

    const data = topCandidates.map((c, idx) => {
        const normEff = (c.predicted_activity - 0.1) / 0.85;
        const normSafe = 1.0 - c.safety_risk;
        const normInter = 1.0 - c.cyp_interaction_risk;
        const normCost = 1.0 - (c.synthetic_cost_score / 10.0);
        const normDrug = 1.0 - (c.rule_violations / 4.0);

        const values = [normEff * 100, normSafe * 100, normInter * 100, normCost * 100, normDrug * 100];
        values.push(values[0]);

        return {
            type: 'scatterpolar',
            r: values,
            theta: [...categories, categories[0]],
            fill: 'toself',
            name: `#${c.optimized_rank} ${c.compound_id}`,
            line: { color: colors[idx % colors.length], width: 2 }
        };
    });

    const layout = {
        title: { text: '<b>Multi-Attribute Lead Profile (Top 4)</b>', font: { color: '#f8fafc', size: 14 } },
        polar: {
            radialaxis: { visible: true, range: [0, 100], color: '#64748b' },
            angularaxis: { color: '#94a3b8' },
            bgcolor: 'transparent'
        },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        margin: { l: 40, r: 40, t: 45, b: 40 },
        height: 400,
        legend: { font: { color: '#cbd5e1' }, orientation: 'h', y: -0.2 }
    };

    Plotly.newPlot('radar-plot', data, layout, { responsive: true, displayModeBar: false });
}

function populateScaffoldFilter(candidates) {
    const scaffolds = [...new Set(candidates.map(c => c.scaffold_type))];
    const select = document.getElementById("filter-scaffold");
    if (select) {
        select.innerHTML = '<option value="all">All Scaffolds</option>';
        scaffolds.forEach(s => {
            const opt = document.createElement("option");
            opt.value = s;
            opt.textContent = s;
            select.appendChild(opt);
        });
    }
}

function filterAndRenderTable() {
    if (!currentPipelineData) return;
    const query = document.getElementById("candidate-search")?.value.toLowerCase() || "";
    const paretoOnly = document.getElementById("filter-pareto")?.checked || false;
    const lipinskiOnly = document.getElementById("filter-lipinski")?.checked || false;
    const scaffoldFilter = document.getElementById("filter-scaffold")?.value || "all";

    let filtered = currentPipelineData.candidates.filter(c => {
        if (paretoOnly && !c.is_pareto_optimal) return false;
        if (lipinskiOnly && c.rule_violations > 1) return false;
        if (scaffoldFilter !== "all" && c.scaffold_type !== scaffoldFilter) return false;
        if (query) {
            const matchId = c.compound_id.toLowerCase().includes(query);
            const matchName = c.compound_name.toLowerCase().includes(query);
            if (!matchId && !matchName) return false;
        }
        return true;
    });

    renderCandidatesTable(filtered);
}

function renderCandidatesTable(candidates) {
    const tbody = document.getElementById("candidates-tbody");
    if (!tbody) return;

    tbody.innerHTML = candidates.map(c => {
        const rankClass = c.optimized_rank === 1 ? 'top-1' : (c.optimized_rank === 2 ? 'top-2' : (c.optimized_rank === 3 ? 'top-3' : ''));
        const paretoStar = c.is_pareto_optimal ? '<span class="star-pareto">⭐ Pareto</span>' : '<span style="color:#64748b;">—</span>';
        const rankDeltaBadge = c.rank_delta > 0 
            ? `<span style="color:#4ade80; font-size:0.75rem;">▲ +${c.rank_delta}</span>`
            : (c.rank_delta < 0 ? `<span style="color:#f87171; font-size:0.75rem;">▼ ${c.rank_delta}</span>` : `<span style="color:#64748b; font-size:0.75rem;">—</span>`);

        return `
            <tr>
                <td><span class="rank-badge ${rankClass}">#${c.optimized_rank}</span> ${rankDeltaBadge}</td>
                <td><b>${c.compound_id}</b><br><span style="color:#94a3b8; font-size:0.8rem;">${c.compound_name}</span></td>
                <td><span style="font-size:0.82rem; background:#1e293b; padding:3px 8px; border-radius:4px;">${c.scaffold_type}</span></td>
                <td><b>${c.predicted_pic50.toFixed(2)}</b></td>
                <td>${c.safety_risk.toFixed(2)}</td>
                <td>${c.cyp_interaction_risk.toFixed(2)}</td>
                <td>${c.synthetic_cost_score.toFixed(1)}/10</td>
                <td><b style="color:#38bdf8;">${c.composite_score.toFixed(1)}</b></td>
                <td>${paretoStar}</td>
                <td><span class="badge" style="background:${c.lipinski_badge === 'green' ? 'rgba(16,185,129,0.15)' : 'rgba(239,68,68,0.15)'}; color:${c.lipinski_badge === 'green' ? '#34d399' : '#f87171'}; font-size:0.75rem;">${c.lipinski_status}</span></td>
                <td>
                    <button class="btn-action" onclick="inspectCandidateById('${c.compound_id}')">Inspect</button>
                    <button class="btn-action" style="border-color:#a855f7; color:#c084fc; margin-left:4px;" onclick="simulateCandidateDocking('${c.compound_id}')">Simulate Docking</button>
                </td>
            </tr>
        `;
    }).join('');
}

window.inspectCandidateById = function(compoundId) {
    if (!currentPipelineData) return;
    const cand = currentPipelineData.candidates.find(c => c.compound_id === compoundId);
    if (cand) inspectCandidate(cand);
};

window.simulateCandidateDocking = function(compoundId) {
    if (!currentPipelineData || !dnaSim) return;
    const cand = currentPipelineData.candidates.find(c => c.compound_id === compoundId);
    if (cand) {
        // Switch to mutant if not already
        if (!dnaSim.isMutant) {
            dnaSim.toggleMutation(true);
            const btn = document.getElementById("btn-toggle-mutant");
            if (btn) {
                btn.classList.add("active-mutant");
                btn.textContent = "Switch to Wild-Type (Healthy)";
            }
        }
        // Dock drug
        dnaSim.dockDrug(cand);

        // Switch tab to DNA simulation if not active
        document.querySelector('[data-tab="tab-dna"]')?.click();
    }
};

function inspectCandidate(c) {
    const inspector = document.getElementById("candidate-inspector");
    if (!inspector) return;

    const explainHtml = (c.explanations || []).map(exp => `
        <div style="font-size:0.84rem; margin-bottom:4px; display:flex; justify-content:space-between;">
            <span style="color:#94a3b8;">• ${exp.feature.replace(/_/g, ' ').toUpperCase()}</span>
            <span style="color:#38bdf8; font-weight:600;">Impact: +${(exp.contribution_score * 100).toFixed(1)}%</span>
        </div>
    `).join('') || '<span style="color:#64748b; font-size:0.84rem;">Feature attribution available for top candidates.</span>';

    inspector.innerHTML = `
        <div class="card" style="border-color:#3b82f6;">
            <div class="card-title">
                <span>🔍 Compound Profile: <b>${c.compound_id}</b> (${c.compound_name})</span>
                <span class="badge" style="background:#1e293b; color:#38bdf8;">Rank #${c.optimized_rank}</span>
            </div>
            <div class="grid-2">
                <div>
                    ${c.svg || `<div style="height:190px; background:#172138; border-radius:8px; display:flex; align-items:center; justify-content:center; color:#64748b;">Molecular Diagram (${c.scaffold_type})</div>`}
                    <div style="font-family:monospace; font-size:0.75rem; color:#64748b; margin-top:8px; word-break:break-all;">SMILES: ${c.smiles}</div>
                </div>
                <div>
                    <h5 style="color:#f8fafc; font-size:0.92rem; margin-bottom:8px;">Physicochemical & ADMET Properties:</h5>
                    <table style="width:100%; font-size:0.84rem; margin-bottom:12px;">
                        <tr><td style="color:#94a3b8;">Molecular Weight:</td><td><b>${c.molecular_weight.toFixed(1)} Da</b></td><td style="color:#94a3b8;">LogP:</td><td><b>${c.logp ? c.logp.toFixed(2) : 'N/A'}</b></td></tr>
                        <tr><td style="color:#94a3b8;">H-Bond Donors:</td><td><b>${c.hbd}</b></td><td style="color:#94a3b8;">H-Bond Acceptors:</td><td><b>${c.hba}</b></td></tr>
                        <tr><td style="color:#94a3b8;">TPSA:</td><td><b>${c.tpsa ? c.tpsa.toFixed(1) : 'N/A'} Å²</b></td><td style="color:#94a3b8;">Rotatable Bonds:</td><td><b>${c.rotatable_bonds}</b></td></tr>
                        <tr><td style="color:#94a3b8;">Safety Risk:</td><td><b>${c.safety_risk.toFixed(2)}</b></td><td style="color:#94a3b8;">CYP450 Risk:</td><td><b>${c.cyp_interaction_risk.toFixed(2)}</b></td></tr>
                    </table>
                    <h5 style="color:#f8fafc; font-size:0.92rem; margin-bottom:6px;">Random Forest Explainability (Why it ranked high):</h5>
                    ${explainHtml}
                    <div style="margin-top:14px;">
                        <button class="btn-primary" style="width:100%;" onclick="simulateCandidateDocking('${c.compound_id}')">🚀 Simulate Target Inhibition Docking</button>
                    </div>
                </div>
            </div>
        </div>
    `;
}

function downloadCSV() {
    if (!currentPipelineData) return;
    const candidates = currentPipelineData.candidates;
    const headers = ["Rank", "Compound_ID", "Name", "Scaffold", "Pred_pIC50", "Safety_Risk", "CYP_Risk", "Cost_Score", "Composite_Score", "Desirability", "Rule_Violations", "Is_Pareto", "SMILES"];
    const rows = candidates.map(c => [
        c.optimized_rank,
        c.compound_id,
        `"${c.compound_name}"`,
        c.scaffold_type,
        c.predicted_pic50,
        c.safety_risk,
        c.cyp_interaction_risk,
        c.synthetic_cost_score,
        c.composite_score,
        c.desirability_index,
        c.rule_violations,
        c.is_pareto_optimal ? "YES" : "NO",
        c.smiles
    ]);

    const csvContent = [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `OptiDrug_${currentPipelineData.target.target_id}_Candidates.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function showLoading(show) {
    const overlay = document.getElementById("loading-overlay");
    if (overlay) overlay.style.display = show ? "flex" : "none";
}
