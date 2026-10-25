"""
OptiDrug - Chemical Structure & Molecular Descriptors Utility
Calculates Lipinski's Rule of 5 and Veber rules, assesses drug-likeness,
and generates chemical structure SVG previews for candidate compound cards.
"""

from typing import Dict, Any, Tuple
import html

def assess_lipinski_rule_of_5(mw: float, logp: float, hbd: int, hba: int) -> Dict[str, Any]:
    """
    Evaluates Lipinski's Rule of 5:
      - Molecular Weight <= 500 Da
      - LogP <= 5.0
      - Hydrogen Bond Donors <= 5
      - Hydrogen Bond Acceptors <= 10
    """
    violations = []
    if mw > 500.0:
        violations.append(f"MW ({mw:.1f} Da > 500)")
    if logp > 5.0:
        violations.append(f"LogP ({logp:.2f} > 5.0)")
    if hbd > 5:
        violations.append(f"HBD ({hbd} > 5)")
    if hba > 10:
        violations.append(f"HBA ({hba} > 10)")
        
    num_violations = len(violations)
    return {
        "num_violations": num_violations,
        "is_lipinski_compliant": num_violations <= 1,
        "violations": violations,
        "badge_color": "green" if num_violations == 0 else ("orange" if num_violations == 1 else "red"),
        "status": "Pass (0 violations)" if num_violations == 0 else (f"Alert ({num_violations} violation)" if num_violations == 1 else f"Fail ({num_violations} violations)")
    }

def assess_veber_rules(rotatable_bonds: int, tpsa: float) -> Dict[str, Any]:
    """
    Evaluates Veber Rules (Oral Bioavailability):
      - Rotatable Bonds <= 10
      - Polar Surface Area (TPSA) <= 140 Å²
    """
    violations = []
    if rotatable_bonds > 10:
        violations.append(f"Rotatable Bonds ({rotatable_bonds} > 10)")
    if tpsa > 140.0:
        violations.append(f"TPSA ({tpsa:.1f} > 140 Å²)")
        
    num_violations = len(violations)
    return {
        "num_violations": num_violations,
        "is_veber_compliant": num_violations == 0,
        "violations": violations,
        "badge_color": "green" if num_violations == 0 else "red",
        "status": "Pass" if num_violations == 0 else f"Fail ({num_violations} violations)"
    }

def generate_compound_svg(compound_id: str, compound_name: str, scaffold: str, mw: float, logp: float) -> str:
    """
    Generates a stylized SVG molecular structure diagram for the candidate compound card.
    Provides clear visual representation even without external C++ RDKit binaries.
    """
    safe_name = html.escape(compound_name)
    safe_id = html.escape(compound_id)
    safe_scaffold = html.escape(scaffold)
    
    # Stylized pharmacophore skeleton representations based on scaffold type
    scaffold_paths = {
        "Quinazoline": """
            <polygon points="120,60 160,83 160,129 120,152 80,129 80,83" fill="none" stroke="#2563eb" stroke-width="3.5"/>
            <polygon points="160,83 200,60 240,83 240,129 200,152 160,129" fill="none" stroke="#2563eb" stroke-width="3.5"/>
            <circle cx="160" cy="83" r="5" fill="#3b82f6"/>
            <circle cx="200" cy="152" r="5" fill="#3b82f6"/>
            <line x1="240" y1="83" x2="275" y2="65" stroke="#ef4444" stroke-width="3"/>
            <line x1="80" y1="83" x2="45" y2="65" stroke="#10b981" stroke-width="3"/>
            <text x="280" y="68" fill="#ef4444" font-family="sans-serif" font-size="12" font-weight="bold">NH</text>
            <text x="30" y="68" fill="#10b981" font-family="sans-serif" font-size="12" font-weight="bold">O</text>
        """,
        "Pyrrolo[2,3-b]pyridine": """
            <polygon points="110,65 150,85 150,130 110,150 70,130 70,85" fill="none" stroke="#8b5cf6" stroke-width="3.5"/>
            <polygon points="150,85 190,98 175,138 150,130" fill="none" stroke="#8b5cf6" stroke-width="3.5"/>
            <circle cx="150" cy="85" r="5" fill="#a855f7"/>
            <line x1="190" y1="98" x2="225" y2="85" stroke="#f59e0b" stroke-width="3"/>
            <line x1="70" y1="130" x2="40" y2="150" stroke="#06b6d4" stroke-width="3"/>
            <text x="230" y="90" fill="#f59e0b" font-family="sans-serif" font-size="12" font-weight="bold">Cl</text>
            <text x="25" y="160" fill="#06b6d4" font-family="sans-serif" font-size="12" font-weight="bold">SO₂</text>
        """,
        "Triazolopiperazine": """
            <polygon points="100,70 145,70 165,110 145,150 100,150 80,110" fill="none" stroke="#059669" stroke-width="3.5"/>
            <polygon points="165,110 205,95 230,120 205,145 165,110" fill="none" stroke="#059669" stroke-width="3.5"/>
            <line x1="80" y1="110" x2="45" y2="110" stroke="#3b82f6" stroke-width="3"/>
            <circle cx="205" cy="95" r="5" fill="#10b981"/>
            <circle cx="230" cy="120" r="5" fill="#10b981"/>
            <text x="18" y="115" fill="#3b82f6" font-family="sans-serif" font-size="12" font-weight="bold">NH₂</text>
            <text x="240" y="125" fill="#ef4444" font-family="sans-serif" font-size="12" font-weight="bold">CF₃</text>
        """
    }
    
    # Default skeleton for other scaffolds
    default_skeleton = """
        <polygon points="120,60 165,85 165,135 120,160 75,135 75,85" fill="none" stroke="#4f46e5" stroke-width="3.5"/>
        <line x1="165" y1="85" x2="205" y2="65" stroke="#4f46e5" stroke-width="3.5"/>
        <line x1="205" y1="65" x2="245" y2="85" stroke="#4f46e5" stroke-width="3.5"/>
        <line x1="245" y1="85" x2="245" y2="135" stroke="#4f46e5" stroke-width="3.5"/>
        <line x1="245" y1="135" x2="205" y2="155" stroke="#4f46e5" stroke-width="3.5"/>
        <line x1="205" y1="155" x2="165" y2="135" stroke="#4f46e5" stroke-width="3.5"/>
        <circle cx="205" cy="65" r="5" fill="#6366f1"/>
        <line x1="75" y1="85" x2="40" y2="65" stroke="#10b981" stroke-width="3"/>
        <line x1="245" y1="85" x2="280" y2="70" stroke="#ef4444" stroke-width="3"/>
        <text x="25" y="68" fill="#10b981" font-family="sans-serif" font-size="12" font-weight="bold">R₁</text>
        <text x="288" y="75" fill="#ef4444" font-family="sans-serif" font-size="12" font-weight="bold">R₂</text>
    """
    
    path_data = scaffold_paths.get(scaffold, default_skeleton)
    
    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 200" width="100%" height="190" style="background: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0;">
        <!-- Header -->
        <rect x="0" y="0" width="320" height="34" fill="#1e293b" rx="8" ry="8"/>
        <rect x="0" y="24" width="320" height="10" fill="#1e293b"/>
        <text x="12" y="22" fill="#ffffff" font-family="system-ui, -apple-system, sans-serif" font-size="12" font-weight="bold">{safe_id} - {safe_name}</text>
        <text x="308" y="22" text-anchor="end" fill="#94a3b8" font-family="system-ui, -apple-system, sans-serif" font-size="10">{safe_scaffold}</text>
        
        <!-- Chemical Skeleton -->
        <g transform="translate(0, 10)">
            {path_data}
        </g>
        
        <!-- Footer Info -->
        <rect x="0" y="170" width="320" height="30" fill="#f1f5f9" rx="0"/>
        <text x="12" y="189" fill="#475569" font-family="system-ui, -apple-system, sans-serif" font-size="11">MW: <b>{mw:.1f}</b></text>
        <text x="120" y="189" fill="#475569" font-family="system-ui, -apple-system, sans-serif" font-size="11">LogP: <b>{logp:.2f}</b></text>
        <text x="220" y="189" fill="#2563eb" font-family="system-ui, -apple-system, sans-serif" font-size="11" font-weight="600">OptiDrug Candidate</text>
    </svg>
    """
    return svg
