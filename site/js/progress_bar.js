/**
 *
 * @param {Array<[number, number, string]>} steps
 */
function addProgressBars(steps) {
  let progress_bar_html = "";
  for (let index = 0; index < steps.length; index++) {
    const now = steps[index][0];
    const step = steps[index][1];
    const bg_class = steps[index][2];
    const as_width = (now * 100) / step;
    progress_bar_html += `<div class="progress-bar ${bg_class} status-bar" role="progressbar" style="width: ${as_width}%" aria-valuenow="0" aria-valuemax="${step}"></div>`;
  }
  document.getElementsByClassName("progress")[0].innerHTML = progress_bar_html;
}
const bars = document.getElementsByClassName("status-bar");
/**
 *
 * @param {number} index
 * @param {number} goal_percentage
 * @param {number} incr
 * @param {number} waitTime
 */

async function animate_progress(
  index,
  goal_percentage,
  incr = 5,
  waitTime = 100
) {
  const values = getBarValues(index);
  let now = values[0];
  if (now >= goal_percentage) {
    return;
  }
  let max = values[1];

  setBarActive(index, true);
  while (now + incr < goal_percentage) {
    now += incr;
    setProgress(index, now);
    await new Promise((r) => setTimeout(r, waitTime));
  }
  setProgress(index, goal_percentage);
  await new Promise((r) => setTimeout(r, 500));
  if (goal_percentage == 100) {
    setBarActive(index, false);
  }
}

function setBarActive(index, active) {
  if (active) {
    bars[index].classList.add("progress-bar-striped");
    bars[index].classList.add("progress-bar-animated");
  } else {
    bars[index].classList.remove("progress-bar-striped");
    bars[index].classList.remove("progress-bar-animated");
  }
}

/**
 *
 * @param {number} index
 * @returns {Array<number>}
 */
function getBarValues(index) {
  return [
    parseInt(bars[index].getAttribute("aria-valuenow")),
    parseInt(bars[index].getAttribute("aria-valuemax")),
  ];
}

function setProgress(index, percentage) {
  const bar = bars[index];
  let as_width =
    parseInt(bar.getAttribute("aria-valuemax")) * (percentage / 100);

  // @ts-ignore
  bar.style.width = as_width + "%";
  bar.setAttribute("aria-valuenow", "" + percentage);
}
