/**
 * keiba-ebye Discord Relay Worker
 * Cloudflare Workers cron → HF Hub キューポーリング → Discord送信
 *
 * 環境変数（Cloudflare ダッシュボードで設定）:
 *   HF_TOKEN                   : HuggingFace APIトークン（read/write権限）
 *   HF_REPO_ID                 : "ebi44323/keiba-ebye-models"
 *   DISCORD_WEBHOOK_URL        : 予想通知用 Webhook URL
 *   DISCORD_REVIEW_WEBHOOK_URL : 振り返り用 Webhook URL（未設定時は予想用を流用）
 *
 * Cron設定: * * * * *（毎分）
 */

const QUEUE_FILE = 'discord_queue.json';

export default {
  // ヘルスチェック用（GETのみ）
  async fetch(request, env) {
    return new Response('keiba-ebye Discord relay running', { status: 200 });
  },

  // Cron trigger: 毎分実行
  async scheduled(event, env, ctx) {
    ctx.waitUntil(processDiscordQueue(env));
  },
};

async function processDiscordQueue(env) {
  try {
    // 1. HF Hub からキューを取得
    const queueUrl = `https://huggingface.co/datasets/${env.HF_REPO_ID}/resolve/main/${QUEUE_FILE}?nocache=${Date.now()}`;
    const resp = await fetch(queueUrl, {
      headers: { 'Authorization': `Bearer ${env.HF_TOKEN}` },
    });
    if (!resp.ok) {
      console.error(`Queue fetch failed: ${resp.status}`);
      return;
    }

    let queue;
    try {
      queue = await resp.json();
    } catch {
      console.error('Queue JSON parse failed');
      return;
    }

    // 2. 未送信エントリを抽出
    const unsent = queue.filter(q => !q.sent);
    if (unsent.length === 0) return;

    // 3. Discord に送信
    let updated = false;
    for (const entry of unsent) {
      const webhookUrl = entry.channel === 'review'
        ? (env.DISCORD_REVIEW_WEBHOOK_URL || env.DISCORD_WEBHOOK_URL)
        : env.DISCORD_WEBHOOK_URL;

      if (!webhookUrl) continue;

      const discordResp = await fetch(webhookUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content: (entry.content || '').slice(0, 2000),
          username: entry.username || 'keiba-ebye',
        }),
      });

      if (discordResp.ok) {
        entry.sent = true;
        updated = true;
        console.log(`Sent: ${entry.id}`);
      } else {
        console.error(`Discord send error ${entry.id}: ${discordResp.status}`);
      }
    }

    // 4. 送信済みフラグを HF Hub に書き戻す
    if (updated) {
      await uploadToHFHub(env, QUEUE_FILE, JSON.stringify(queue, null, 2));
    }
  } catch (err) {
    console.error(`processDiscordQueue error: ${err}`);
  }
}

async function uploadToHFHub(env, path, content) {
  const lines = [
    JSON.stringify({ key: 'header', value: { summary: 'Discord queue update' } }),
    JSON.stringify({ key: 'file', value: { path, encoding: 'utf-8', content } }),
  ].join('\n');

  const resp = await fetch(
    `https://huggingface.co/api/datasets/${env.HF_REPO_ID}/commit/main`,
    {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${env.HF_TOKEN}`,
        'Content-Type': 'application/x-ndjson',
      },
      body: lines,
    }
  );
  if (!resp.ok) {
    const text = await resp.text();
    console.error(`HF Hub upload failed: ${resp.status} ${text}`);
  }
}
