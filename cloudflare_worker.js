/**
 * keiba-ebye Discord Relay Worker
 * HuggingFace Spaces → Cloudflare Workers → Discord Webhook
 *
 * 環境変数（Cloudflare ダッシュボードで設定）:
 *   AUTH_TOKEN                  : 認証トークン（任意の文字列）
 *   DISCORD_WEBHOOK_URL         : 予想通知用 Webhook URL
 *   DISCORD_REVIEW_WEBHOOK_URL  : 振り返り用 Webhook URL（未設定時は予想用を流用）
 */

export default {
  async fetch(request, env) {
    if (request.method !== 'POST') {
      return new Response('Method Not Allowed', { status: 405 });
    }

    // 認証チェック
    const auth = request.headers.get('Authorization') || '';
    if (!env.AUTH_TOKEN || auth !== `Bearer ${env.AUTH_TOKEN}`) {
      return new Response('Unauthorized', { status: 401 });
    }

    let body;
    try {
      body = await request.json();
    } catch {
      return new Response('Bad Request', { status: 400 });
    }

    // channel に応じて送信先 Webhook URL を選択
    const channel = body.channel || 'prediction';
    const webhookUrl = channel === 'review'
      ? (env.DISCORD_REVIEW_WEBHOOK_URL || env.DISCORD_WEBHOOK_URL)
      : env.DISCORD_WEBHOOK_URL;

    if (!webhookUrl) {
      return new Response('Webhook URL not configured', { status: 500 });
    }

    const discordResp = await fetch(webhookUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        content: (body.content || '').slice(0, 2000),
        username: body.username || 'keiba-ebye',
      }),
    });

    if (discordResp.ok) {
      return new Response('OK', { status: 200 });
    }
    const errText = await discordResp.text();
    return new Response(`Discord error: ${errText}`, { status: discordResp.status });
  },
};
