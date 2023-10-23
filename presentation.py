from typing import List

from jinja2 import Template
from markdown import markdown


def open_html_file_in_browser(filename: str):
    import webbrowser
    webbrowser.open(filename)


def save_html_to_file(html: str, filename: str):
    with open(filename, 'w') as f:
        f.write(html)


def generate_decision_report_as_html(criteria: List[any], alternatives: List[any], goal: str) -> str:
    sorted_alternatives = sorted(alternatives, key=lambda x: x['score'], reverse=True)

    template_str = '''<!DOCTYPE html>
<html>
<head>
  <title>Decision Report</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/semantic-ui/2.5.0/semantic.min.css" integrity="sha512-KXol4x3sVoO+8ZsWPFI/r5KBVB/ssCGB5tsv2nVOKwLg33wTFP3fmnXa47FdSVIshVTgsYk/1734xSk9aFIa4A==" crossorigin="anonymous" referrerpolicy="no-referrer" />
  <script src="https://code.jquery.com/jquery-3.1.1.min.js"
          integrity="sha256-hVVnYaiADRTO2PzUGmuLJr8BLUSjGIZsDYGmIJLv2b8="
          crossorigin="anonymous"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/semantic-ui/2.5.0/semantic.min.js" integrity="sha512-Xo0Jh8MsOn72LGV8kU5LsclG7SUzJsWGhXbWcYs2MAmChkQzwiW/yTQwdJ8w6UA9C6EVG18GHb/TrYpYCjyAQw==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
  <style>
    li, p, tbody {
      line-height: 1.6;
      font-size: 20px;
    }
    body {
        background-color: lightblue;
    }
    body > .ui.container {
       padding: 8em;
    }
  </style>
</head>
<body>
  <div class="ui container" style="margin-top: 20px;">
    <div class="ui raised very padded text segment">
  <h1 class="ui header">Goal</h1>
  <p>{{ goal }}</p>
  <div class="ui divider"></div>

  <h2 class="ui header">Alternatives</h2>
   <table class="ui celled sortable table">
    <thead>
      <tr>
        <th>Alternative</th>
        {% for criterion in criteria %}
            <th>{{ criterion['name'] }}</th>
        {% endfor %}
        <th>Score</th>
      </tr>
    </thead>
    <tbody>
      {% for alternative in sorted_alternatives %}
          <tr {% if loop.first %}class="positive"{% endif %}>
                <td>{{ alternative['name'] }}</td>
            {% for criterion in criteria %}
                <td>{{ alternative['criteria_data'][criterion['name']]['aggregated']['label'] }}</td>
            {% endfor %}
            <td>{{ "{:.0%}".format(alternative['score']) }}</td>
          </tr>
      {% endfor %}
    </tbody>
   </table>

  <h2 class="ui header">Criteria</h2>
    <table class="ui celled sortable table">
    <thead>
      <tr>
        <th>Criterion</th>
        <th>Scale</th>
      </tr>
    </thead>
    <tbody>
      {% for criterion in criteria %}
      <tr>
        <td>{{ criterion['name'] }}</td>
        <td><ol class="ui list">
          {% for scale_value in criterion['scale'] %}
              <li>{{ scale_value }}</li>
          {% endfor %}
        </ol></td>
      </tr>
      {% endfor %}
    </tbody>
  </table>

  <h2 class="ui header">Full Research Findings</h2>
    {% for alternative in sorted_alternatives %}
        <h3 class="ui header">{{ alternative['name'] }}</h3>
        {% for criterion in criteria %}
            <h4 class="ui header">{{ criterion['name'] }}</h4>
            <p>{{ markdown(alternative['criteria_data'][criterion['name']]['aggregated']['findings']) }}</p>
            <p><strong>Assigned label: {{ alternative['criteria_data'][criterion['name']]['aggregated']['label'] }}</strong></p>
        {% endfor %}
    {% endfor %}
</div>
</body>
</html>'''

    template = Template(template_str)
    return template.render(criteria=criteria, sorted_alternatives=sorted_alternatives, goal=goal, markdown=markdown)
