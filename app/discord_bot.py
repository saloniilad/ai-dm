"""Discord bot: one shared campaign per channel.
   pip install -r requirements-discord.txt ; export DISCORD_TOKEN=... ; python -m app.discord_bot
   Enable the 'Message Content Intent' for your bot in the Discord developer portal.
   Commands: !join [name]  !do <action>  !state  !recap"""
import asyncio
import os

import discord
from discord.ext import commands

from dm.session import Session

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)
sessions: dict[int, Session] = {}
locks: dict[int, asyncio.Lock] = {}


def session_for(ctx) -> Session:
    cid = ctx.channel.id
    if cid not in sessions:
        sessions[cid] = Session(f"discord-{cid}")
        locks[cid] = asyncio.Lock()
    return sessions[cid]


def chunks(text, n=1900):
    return [text[i:i + n] for i in range(0, len(text), n)] or ["…"]


@bot.command()
async def join(ctx, *, name: str = None):
    s = session_for(ctx)
    eng = s.join(name or ctx.author.display_name)
    await ctx.send(f"**{eng.player.name}** joins at {eng.player.location}. Use `!do <action>`.")


@bot.command(name="do")
async def do(ctx, *, action: str):
    s = session_for(ctx)
    who = ctx.author.display_name
    async with locks[ctx.channel.id]:
        async with ctx.typing():
            try:
                r = await asyncio.to_thread(s.act, who, action)
            except RuntimeError as e:
                return await ctx.send(str(e))
    for part in chunks(r.text):
        await ctx.send(part)
    st = s.state(who)
    await ctx.send(f"`HP {st['hp']}/{st['max_hp']} | Gold {st['gold']} | {st['location']}`")


@bot.command()
async def state(ctx):
    s = session_for(ctx)
    who = ctx.author.display_name
    if who not in s.engines:
        return await ctx.send("Use `!join` first.")
    await ctx.send(f"```{s.state(who)}```")


@bot.command()
async def recap(ctx):
    s = session_for(ctx)
    text = await asyncio.to_thread(s.recap)
    for part in chunks(text):
        await ctx.send(part)


if __name__ == "__main__":
    bot.run(os.environ["DISCORD_TOKEN"])
