import markdown
import re

with open("blog.md", "r", encoding="utf-8") as f:
    md_content = f.read()

html_content = markdown.markdown(md_content, extensions=['fenced_code', 'tables'])

# Correct the image paths to point to /plots/
html_content = html_content.replace('src="combined_results.png"', 'src="/plots/combined_results.png"')
html_content = html_content.replace('src="loss_curve.png"', 'src="/plots/loss_curve.png"')

blog_screen = f"""
    <div id="screen-blog" class="screen">
      <div class="card" style="padding: 48px; border: none; box-shadow: var(--shadow-md); max-width: 900px; margin: 0 auto; line-height: 1.8; font-size: 16px; color: var(--text-primary);">
        <style>
          #screen-blog h1 {{ font-size: 36px; font-weight: 800; margin-bottom: 24px; color: #0f172a; line-height: 1.2; letter-spacing: -0.02em; }}
          #screen-blog h2 {{ font-size: 26px; font-weight: 700; margin-top: 48px; margin-bottom: 20px; color: #1e293b; border-bottom: 1px solid var(--border); padding-bottom: 10px; letter-spacing: -0.01em; }}
          #screen-blog h3 {{ font-size: 20px; font-weight: 700; margin-top: 32px; margin-bottom: 16px; color: #334155; }}
          #screen-blog p {{ margin-bottom: 24px; color: #475569; }}
          #screen-blog blockquote {{ border-left: 4px solid var(--indigo); padding-left: 20px; margin: 32px 0; background: var(--indigo-soft); padding: 20px; border-radius: 0 8px 8px 0; color: var(--indigo); font-weight: 500; font-size: 18px; line-height: 1.6; }}
          #screen-blog pre {{ background: #1e293b; color: #f8fafc; padding: 20px; border-radius: 12px; overflow-x: auto; font-family: 'DM Mono', monospace; font-size: 14px; margin-bottom: 32px; box-shadow: inset 0 2px 4px rgba(0,0,0,0.2); line-height: 1.5; }}
          #screen-blog code {{ font-family: 'DM Mono', monospace; background: var(--gray-100); padding: 3px 6px; border-radius: 4px; font-size: 14px; color: #ef4444; border: 1px solid var(--border); }}
          #screen-blog pre code {{ background: transparent; padding: 0; color: inherit; border: none; }}
          #screen-blog ul, #screen-blog ol {{ margin-bottom: 32px; padding-left: 24px; color: #475569; }}
          #screen-blog li {{ margin-bottom: 12px; }}
          #screen-blog table {{ width: 100%; border-collapse: collapse; margin-bottom: 32px; font-size: 14px; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
          #screen-blog th, #screen-blog td {{ padding: 14px 16px; border: 1px solid var(--border); text-align: left; }}
          #screen-blog th {{ background: #f8fafc; font-weight: 600; color: #334155; text-transform: uppercase; font-size: 12px; letter-spacing: 0.05em; }}
          #screen-blog td {{ background: #ffffff; }}
          #screen-blog img {{ max-width: 100%; border-radius: 12px; margin: 32px 0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); border: 1px solid var(--border); }}
          #screen-blog em {{ color: #64748b; font-style: italic; }}
          #screen-blog strong {{ font-weight: 700; color: #0f172a; }}
        </style>
        {html_content}
      </div>
    </div>
"""

with open("fleet_bench_ui.html", "r", encoding="utf-8") as f:
    ui_html = f.read()

# 1. Add Navigation Item (Idempotent)
if 'onclick="showScreen(\'blog\')"' not in ui_html:
    nav_insert = """    <div class="nav-item" onclick="showScreen('training')"><div class="nav-dot"></div>Training Results</div>
    <div class="nav-item" onclick="showScreen('blog')"><div class="nav-dot"></div>Project Blog</div>"""
    ui_html = ui_html.replace("""    <div class="nav-item" onclick="showScreen('training')"><div class="nav-dot"></div>Training Results</div>""", nav_insert)

# 2. Add Blog Screen (Replace existing if present)
if '<div id="screen-blog"' in ui_html:
    # Use regex to replace the entire old screen-blog div
    ui_html = re.sub(r'<div id="screen-blog".*?</div>\s*</div>\s*</div>', blog_screen.strip(), ui_html, flags=re.DOTALL)
else:
    ui_html = ui_html.replace("  </main>", blog_screen + "\n  </main>")

# 3. Add Screen Title (Idempotent)
if 'blog: ["Project Blog"' not in ui_html:
    title_insert = """  training: ["Training Results", "GRPO training · Qwen2.5-1.5B-Instruct · HF Jobs T4 · 30 episodes · easy_fleet"],
  blog: ["Project Blog", "Deep dive into the architecture and governance philosophy"],"""
    ui_html = ui_html.replace("""  training: ["Training Results", "GRPO training · Qwen2.5-1.5B-Instruct · HF Jobs T4 · 30 episodes · easy_fleet"],""", title_insert)

# 4. Add Sidebar Logic (Idempotent)
if "if (screenId === 'blog') return txt.includes('blog');" not in ui_html:
    show_screen_logic = """    if (screenId === 'training') return txt.includes('results');
    if (screenId === 'blog') return txt.includes('blog');"""
    ui_html = ui_html.replace("""    if (screenId === 'training') return txt.includes('results');""", show_screen_logic)

with open("fleet_bench_ui.html", "w", encoding="utf-8") as f:
    f.write(ui_html)
print("Blog updated successfully (idempotent)!")
