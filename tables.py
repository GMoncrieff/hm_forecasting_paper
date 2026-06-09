"""Table-generating functions (reportlab PDFs). Outputs to config.OUTPUT_DIR."""
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

import config

def table_s1():
    """Supplementary Table S1: model covariates -> landscape PDF."""
    OUT = config.OUTPUT_DIR
    OUT.mkdir(parents=True, exist_ok=True)
    XLSX = config.PATHS['covariates_xlsx']
    OUT_PDF = OUT / 'Supplementary_Table_S1_covariates.pdf'

    TABLE_LABEL = 'Table S1'

    # Columns to show (the source `Type` column is consumed as a group subheading).
    DISPLAY_COLS = [
        'Covariate', 'Source dataset', 'Native spatial resolution',
        'Native temporal resolution', 'Time period used', 'Units',
        'Citation', 'Link / DOI',
    ]
    COL_WIDTHS = [1.55, 2.05, 0.95, 1.05, 1.15, 0.80, 1.05, 1.40]  # inches, sum=10.0
    LINK_COL = DISPLAY_COLS.index('Link / DOI')

    # Order groups appear in the table; label shown on the spanning subheading row.
    GROUP_LABEL = {
        'Static': 'Static covariates',
        'Dynamic': 'Dynamic covariates',
        'Static (learned)': 'Static (learned) covariates',
    }

    CELL = ParagraphStyle('cell', fontName='Times-Roman', fontSize=8.5, leading=10)
    HEAD = ParagraphStyle('head', fontName='Times-Bold', fontSize=8.5, leading=10)
    LINK = ParagraphStyle('link', fontName='Times-Roman', fontSize=8.5, leading=10,
                          textColor=colors.HexColor('#1A56A0'))


    def markup(text: str) -> str:
        """XML-escape a cell value, then restore the markup we intend to keep."""
        s = escape('' if pd.isna(text) else str(text).strip())
        s = s.replace('⁻¹', '<super>-1</super>')
        return s


    def load() -> tuple[str, pd.DataFrame, str]:
        """Return (description, data-with-Type, notes) parsed from the workbook."""
        raw = pd.read_excel(XLSX, header=None)
        col0 = raw.iloc[:, 0].astype('string')
        hdr = col0.eq('Covariate').idxmax()                 # header row index
        description = str(raw.iloc[hdr - 2, 0]).strip()     # row above the blank
        notes_idx = col0.str.startswith('Notes', na=False).idxmax()
        notes = str(raw.iloc[notes_idx, 0]).strip()

        df = raw.iloc[hdr + 1:notes_idx].copy()
        df.columns = raw.iloc[hdr].tolist()
        df = df.dropna(how='all').reset_index(drop=True)
        return description, df, notes


    def build_table(df: pd.DataFrame) -> Table:
        header = [Paragraph(markup(c), HEAD) for c in DISPLAY_COLS]
        rows = [header]

        style = [
            ('LINEABOVE',     (0, 0), (-1, 0), 1.0, colors.black),
            ('LINEBELOW',     (0, 0), (-1, 0), 1.0, colors.black),
            ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING',   (0, 0), (-1, -1), 4),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 4),
            ('TOPPADDING',    (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]

        group_rows = []
        for gtype in GROUP_LABEL:
            grp = df[df['Type'].astype('string').str.strip() == gtype]
            if grp.empty:
                continue
            rows.append([GROUP_LABEL[gtype], '', '', '', '', '', '', ''])
            group_rows.append(len(rows) - 1)
            for _, r in grp.iterrows():
                cells = []
                for j, col in enumerate(DISPLAY_COLS):
                    if j == LINK_COL and not pd.isna(r[col]):
                        url = str(r[col]).strip()
                        cells.append(Paragraph(
                            f'<link href="{escape(url)}">{escape(url)}</link>', LINK))
                    else:
                        cells.append(Paragraph(markup(r[col]), CELL))
                rows.append(cells)

        for ri in group_rows:
            style += [
                ('SPAN',          (0, ri), (-1, ri)),
                ('FONTNAME',      (0, ri), (-1, ri), 'Times-Bold'),
                ('FONTSIZE',      (0, ri), (-1, ri), 9),
                ('ALIGN',         (0, ri), (-1, ri), 'LEFT'),
                ('BACKGROUND',    (0, ri), (-1, ri), colors.HexColor('#EFEFEF')),
                ('TOPPADDING',    (0, ri), (-1, ri), 4),
                ('BOTTOMPADDING', (0, ri), (-1, ri), 4),
            ]
        style.append(('LINEBELOW', (0, -1), (-1, -1), 1.0, colors.black))

        table = Table(rows, colWidths=[w * inch for w in COL_WIDTHS], repeatRows=1)
        table.setStyle(TableStyle(style))
        return table


    def main() -> None:
        description, df, notes = load()

        doc = SimpleDocTemplate(
            str(OUT_PDF), pagesize=landscape(letter),
            leftMargin=0.5 * inch, rightMargin=0.5 * inch,
            topMargin=0.5 * inch, bottomMargin=0.5 * inch,
            title='Supplementary Table S1. Model covariates',
        )

        caption_style = ParagraphStyle(
            'caption', fontName='Times-Roman', fontSize=11, leading=14, spaceAfter=8)
        notes_style = ParagraphStyle(
            'notes', fontName='Times-Roman', fontSize=8, leading=10, spaceBefore=8)

        story = [
            Paragraph(f'<b>{TABLE_LABEL}.</b> {markup(description)}', caption_style),
            build_table(df),
            Spacer(1, 4),
            Paragraph(markup(notes), notes_style),
        ]
        doc.build(story)
        print(f'Saved {OUT_PDF}  ({len(df)} covariates)')

    main()


def tables_s2_s9():
    """Supplementary Tables S2-S9: per-realm ecoregion composition -> PDF.

    Depends on fig10's unprotected_loss_stats.csv; computes it (via fig10) if absent.
    """
    OUT = config.OUTPUT_DIR
    OUT.mkdir(parents=True, exist_ok=True)
    STATS_CSV = OUT / 'unprotected_loss_stats.csv'
    OUT_DIR = OUT / 'realm_tables'
    TABLES_DIR = OUT_DIR
    OUT_PDF = OUT / 'realm_tables.pdf'
    if not STATS_CSV.exists():
        import figures
        figures.fig10()

    COLUMN_MAP = {
        'BIOME_NAME': 'Biome',
        'ECO_NAME': 'Ecoregion',
        'pct_protected': 'Protected',
        'pct_grey_upper': 'Still Natural 2040 (upper)',
        'pct_grey_central': 'Still Natural 2040 (central)',
        'pct_lost_upper': 'Natural lands loss 2040 (upper)',
        'pct_lost_central': 'Natural Lands loss 2040 (central)',
        'pct_unprot_nonnatural_2020': 'Non-natural 2020',
    }

    PROPORTION_COLS = [
        'Protected',
        'Still Natural 2040 (upper)',
        'Still Natural 2040 (central)',
        'Natural lands loss 2040 (upper)',
        'Natural Lands loss 2040 (central)',
        'Non-natural 2020',
    ]


    def _export_main() -> None:
        if not STATS_CSV.exists():
            raise SystemExit(
                f"Stats CSV not found: {STATS_CSV}\n"
                "Run plot_unprotected_loss.py first to generate it."
            )

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Reading {STATS_CSV.name}…")
        stats = pd.read_csv(STATS_CSV)

        missing = [c for c in COLUMN_MAP if c not in stats.columns]
        if missing:
            raise SystemExit(
                f"Stats CSV is missing required columns: {missing}\n"
                "Re-run plot_unprotected_loss.py to regenerate with the latest schema."
            )

        print(f"  {len(stats)} ecoregions across {stats['REALM'].nunique()} realms")

        for realm, group in stats.groupby('REALM'):
            out = group[list(COLUMN_MAP.keys())].rename(columns=COLUMN_MAP).copy()
            out[PROPORTION_COLS] = (out[PROPORTION_COLS] / 100.0).round(4)
            out = out.sort_values(['Biome', 'Ecoregion']).reset_index(drop=True)

            fname = f"{realm.replace(' ', '_').replace('/', '_')}.csv"
            out_path = OUT_DIR / fname
            out.to_csv(out_path, index=False)
            print(f"  saved {fname}  ({len(out)} rows)")

        print("=== done ===")

    REALM_ORDER = [
        'Afrotropic', 'Antarctica', 'Australasia', 'Indomalayan',
        'Nearctic', 'Neotropic', 'Oceania', 'Palearctic',
    ]

    REALM_ADJECTIVE = {
        'Afrotropic':  'Afrotropical',
        'Antarctica':  'Antarctic',
        'Australasia': 'Australasian',
        'Indomalayan': 'Indomalayan',
        'Nearctic':    'Nearctic',
        'Neotropic':   'Neotropical',
        'Oceania':     'Oceanian',
        'Palearctic':  'Palearctic',
    }

    # The protected mask is hm_static_iucn_strict_1000.tif — IUCN categories I–IV
    # are the conventional "strict" set. Adjust the label if your raster differs.
    IUCN_TEXT = 'IUCN categories I–IV'

    DECIMALS = 3


    def caption_html(table_num: int, realm: str) -> str:
        return (
            f"<b>Table S{table_num+1}.</b> Proportion of {REALM_ADJECTIVE[realm]} "
            f"ecoregions formally protected ({IUCN_TEXT}) in 2020, and the "
            f"projected 2040 status of unprotected lands in upper and central "
            f"scenarios."
        )


    def fmt(x: float) -> str:
        return f"{x:.{DECIMALS}f}"


    ECO_STYLE = ParagraphStyle(
        'eco_cell', fontName='Times-Roman', fontSize=12, leading=14,
    )


    def build_table(df: pd.DataFrame) -> Table:
        """Build a reportlab Table for one realm's ecoregion data."""
        df = df.sort_values(['Biome', 'Ecoregion']).reset_index(drop=True)

        # Two-row header:
        # Row 0 — grouped labels only (over cols 2-3 and 4-5), other cols empty.
        # Row 1 — actual column titles for every column.
        # This keeps the underline below the grouped labels well above the
        # single-row column titles, so nothing is bisected by a line.
        rows = [
            ['', '',
             'Still Natural 2040', '',
             'Natural lands loss 2040', '',
             ''],
            ['Ecoregion', 'Protected',
             'upper', 'central',
             'upper', 'central',
             'Non-natural 2020'],
        ]

        style_cmds: list = [
            # Grouped header spans + underlines (only over the grouped cells)
            ('SPAN',      (2, 0), (3, 0)),
            ('SPAN',      (4, 0), (5, 0)),
            ('LINEBELOW', (2, 0), (3, 0), 0.5, colors.black),
            ('LINEBELOW', (4, 0), (5, 0), 0.5, colors.black),
            # Outer header rules
            ('LINEABOVE', (0, 0), (-1, 0), 1.0, colors.black),
            ('LINEBELOW', (0, 1), (-1, 1), 1.0, colors.black),
            # Fonts
            ('FONTNAME', (0, 0), (-1, 1), 'Times-Bold'),
            ('FONTNAME', (0, 2), (-1, -1), 'Times-Roman'),
            # Sizes / alignment
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('LEADING',  (0, 0), (-1, -1), 14),
            ('VALIGN',   (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN',    (0, 0), (0, -1), 'LEFT'),
            ('ALIGN',    (1, 0), (-1, -1), 'CENTER'),
            # Padding
            ('LEFTPADDING',   (0, 0), (-1, -1), 4),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 4),
            ('TOPPADDING',    (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]

        biome_row_indices = []
        for biome_name, biome_grp in df.groupby('Biome', sort=False):
            rows.append([biome_name, '', '', '', '', '', ''])
            biome_row_indices.append(len(rows) - 1)
            for _, r in biome_grp.iterrows():
                rows.append([
                    Paragraph(r['Ecoregion'], ECO_STYLE),  # Paragraph wraps long names
                    fmt(r['Protected']),
                    fmt(r['Still Natural 2040 (upper)']),
                    fmt(r['Still Natural 2040 (central)']),
                    fmt(r['Natural lands loss 2040 (upper)']),
                    fmt(r['Natural Lands loss 2040 (central)']),
                    fmt(r['Non-natural 2020']),
                ])

        # Bold biome subheading rows that span the whole table.
        for ri in biome_row_indices:
            style_cmds += [
                ('SPAN',     (0, ri), (-1, ri)),
                ('FONTNAME', (0, ri), (-1, ri), 'Times-Bold'),
                ('ALIGN',    (0, ri), (-1, ri), 'LEFT'),
                ('BACKGROUND', (0, ri), (-1, ri), colors.HexColor('#EFEFEF')),
                ('TOPPADDING',    (0, ri), (-1, ri), 4),
                ('BOTTOMPADDING', (0, ri), (-1, ri), 4),
            ]

        style_cmds.append(('LINEBELOW', (0, -1), (-1, -1), 1.0, colors.black))

        col_widths = [
            3.10 * inch,  # Ecoregion (Paragraph wraps long names)
            0.80 * inch,  # Protected
            0.95 * inch,  # Still Natural upper
            0.95 * inch,  # Still Natural central
            1.10 * inch,  # Natural loss upper (group label needs ~2.2in across cols 4-5)
            1.10 * inch,  # Natural loss central
            1.30 * inch,  # Non-natural 2020
        ]
        table = Table(rows, colWidths=col_widths, repeatRows=2)
        table.setStyle(TableStyle(style_cmds))
        return table


    def _build_main() -> None:
        if not TABLES_DIR.exists():
            raise SystemExit(f"Tables directory not found: {TABLES_DIR}")

        doc = SimpleDocTemplate(
            str(OUT_PDF),
            pagesize=landscape(letter),
            leftMargin=0.5 * inch,
            rightMargin=0.5 * inch,
            topMargin=0.5 * inch,
            bottomMargin=0.5 * inch,
            title='Per-realm ecoregion composition tables',
        )

        caption_style = ParagraphStyle(
            'caption',
            fontName='Times-Roman',
            fontSize=12,
            leading=15,
            spaceAfter=8,
        )

        story = []
        table_num = 0
        for realm in REALM_ORDER:
            csv_path = TABLES_DIR / f"{realm}.csv"
            if not csv_path.exists():
                print(f"  warning: {csv_path.name} not found, skipping")
                continue
            df = pd.read_csv(csv_path)
            table_num += 1
            story.append(Paragraph(caption_html(table_num, realm), caption_style))
            story.append(build_table(df))
            if realm != REALM_ORDER[-1]:
                story.append(PageBreak())
            print(f"  Table {table_num}: {realm} ({len(df)} ecoregions)")

        doc.build(story)
        print(f"Saved {OUT_PDF}")

    _export_main()
    _build_main()
