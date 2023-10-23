from typing import List, Dict

from jinja2 import Template
from markdown import markdown
import webbrowser


def open_html_file_in_browser(filename: str):
    webbrowser.open(filename)


def save_html_to_file(html: str, filename: str):
    with open(filename, 'w') as f:
        f.write(html)


def label_to_color(label: str, scale: List[str]) -> str:
    label_index = scale.index(label) + 1
    percentage_to_max = label_index / len(scale)

    if percentage_to_max <= 0.2:
        return 'red'

    if percentage_to_max <= 0.4:
        return 'orange'

    if percentage_to_max <= 0.6:
        return 'yellow'

    if percentage_to_max <= 0.8:
        return 'olive'

    return 'green'


def generate_decision_report_as_html(criteria: List[any], criteria_weights: Dict[str, float], alternatives: List[any],
                                     goal: str) -> str:
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
                <td>
                    <span class="ui large {{ label_to_color(alternative['criteria_data'][criterion['name']]['aggregated']['label'], criterion['scale']) }} label">
                        {{ alternative['criteria_data'][criterion['name']]['aggregated']['label'] }}
                    </span>
                </td>
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
        <th>Weight</th>
      </tr>
    </thead>
    <tbody>
      {% for criterion in criteria %}
      <tr>
        <td>{{ criterion['name'] }}</td>
        <td><ol class="ui list">
          {% for scale_value in criterion['scale'] %}
              <li>
                <div class="ui large basic {{ label_to_color(scale_value, criterion['scale']) }} label">
                    {{ scale_value }}
                </div>
              </li>
          {% endfor %}
        </ol></td>
        <td>{{ "{:.0%}".format(criteria_weights[criterion['name']]) }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>

  <h2 class="ui header">Full Research Findings</h2>
    {% for alternative in sorted_alternatives %}
        <h3 class="ui header">{{ alternative['name'] }}</h3>
        {% for criterion in criteria %}
            <h4 class="ui header">{{ criterion['name'] }}</h4>
            <p>
                <div class="ui basic large {{ label_to_color(alternative['criteria_data'][criterion['name']]['aggregated']['label'], criterion['scale']) }} label">
                    {{ alternative['criteria_data'][criterion['name']]['aggregated']['label'] }}
                </div>
            </p>
            <p>{{ markdown(alternative['criteria_data'][criterion['name']]['aggregated']['findings']) }}</p>
        {% endfor %}
    {% endfor %}
</div>
</body>
</html>'''

    template = Template(template_str)
    return template.render(criteria=criteria,
                           criteria_weights=criteria_weights,
                           sorted_alternatives=sorted_alternatives,
                           goal=goal,
                           markdown=markdown,
                           label_to_color=label_to_color)
