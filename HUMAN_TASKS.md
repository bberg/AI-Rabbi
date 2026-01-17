# Human Tasks for Phase 0 Completion

This document outlines all tasks that require human intervention to complete the Phase 0 implementation. Code changes have been implemented; these are the manual configuration, deployment, and verification steps.

---

## Priority: HIGH (Required for Launch)

### 1. Set Up Plausible Analytics

**Why**: Analytics are essential for tracking user engagement and measuring success metrics.

**Steps**:
1. Go to [Plausible.io](https://plausible.io/) and create an account
2. Add your domain (e.g., `airabbi.com` or your Railway URL)
3. Copy your domain name
4. Edit `templates/index.html` line 21:
   ```html
   <!-- Change this: -->
   <script defer data-domain="YOUR_DOMAIN" src="https://plausible.io/js/script.js"></script>

   <!-- To this (example): -->
   <script defer data-domain="airabbi.com" src="https://plausible.io/js/script.js"></script>
   ```
5. Alternatively, for self-hosted Plausible or other analytics, replace the entire script tag

**Cost**: Free for <10k monthly pageviews, or ~$9/month for unlimited

**Verification**: Visit your site, then check Plausible dashboard for the visit

---

### 2. Deploy pgvector on Railway

**Why**: Replaces Pinecone ($70+/mo) with pgvector ($5-15/mo), eliminates dormancy issues.

**Steps**:

1. **Deploy pgvector template on Railway**:
   - Go to [Railway pgvector template](https://railway.com/deploy/pgvector-latest)
   - Click "Deploy Now"
   - Wait for deployment to complete (~2 minutes)

2. **Get the connection string**:
   - In Railway dashboard, click on the pgvector service
   - Go to "Variables" tab
   - Copy the `DATABASE_URL` value

3. **Set environment variables on your app service**:
   ```
   DATABASE_URL=postgresql://...  (from step 2)
   VECTOR_DB_BACKEND=pgvector
   SECRET_KEY=<generate a random 32+ character string>
   ```

4. **Run the migration script** (see Task #3 below)

5. **Verify the migration** (see Task #4 below)

6. **Remove Pinecone environment variables** (after verification):
   - Delete `PINECONE_API_KEY`
   - Delete `PINECONE_ENV`

**Time estimate**: 30-60 minutes

---

### 3. Run the Database Migration

**Why**: Populates pgvector with Midrash embeddings from the existing pickle files.

**Prerequisites**:
- pgvector deployed on Railway (Task #2)
- `DATABASE_URL` and `OPENAI_API_KEY` set in environment

**Steps**:

1. **Set environment variables locally**:
   ```bash
   export DATABASE_URL="postgresql://..."  # From Railway
   export OPENAI_API_KEY="sk-..."          # Your OpenAI key
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run dry-run first** to verify setup:
   ```bash
   python migrate_to_pgvector.py --dry-run
   ```

   Expected output:
   ```
   Loading source data from output-5sentence_without_embeddings.pkl...
   Found 30000+ records to migrate
   === DRY RUN MODE ===
   Would migrate 30000+ records
   ```

4. **Run the actual migration**:
   ```bash
   python migrate_to_pgvector.py --batch-size 50
   ```

   This will:
   - Generate embeddings for each text segment using OpenAI
   - Upload to pgvector in batches
   - Save checkpoints for resume capability
   - Take approximately 2-4 hours for ~30k records

5. **If interrupted**, resume with:
   ```bash
   python migrate_to_pgvector.py --resume
   ```

**Cost**: ~$1-2 in OpenAI API costs for embedding generation

**Time estimate**: 2-4 hours (mostly waiting)

---

### 4. Verify Migration Success

**Steps**:

1. **Run verification**:
   ```bash
   python migrate_to_pgvector.py --verify
   ```

   Expected output:
   ```
   Verifying migration...
   Source records: 30000+
   Database records: 30000+
   Migration verified successfully!
   ```

2. **Run test query**:
   ```bash
   python migrate_to_pgvector.py --test
   ```

   Expected output:
   ```
   Testing query...
   Query: 'What is the meaning of life?'
   Results: 3 matches

   1. Score: 0.8234
      Source: Sefaria-Export/txt/Midrash/...
      Text: ...relevant passage...
   ```

3. **Test the app locally**:
   ```bash
   export VECTOR_DB_BACKEND=pgvector
   python app.py
   ```

   Visit http://localhost:3000 and try a search

---

### 5. Set Production Environment Variables

**Required variables for Railway**:

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:pass@host:5432/db` |
| `VECTOR_DB_BACKEND` | Set to `pgvector` | `pgvector` |
| `OPENAI_API_KEY` | Your OpenAI API key | `sk-...` |
| `SECRET_KEY` | Random string for Flask sessions | `a1b2c3d4...` (32+ chars) |
| `LOG_PASSWORD` | Admin password for /logs | `your-secure-password` |

**To generate a secure SECRET_KEY**:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## Priority: MEDIUM (Recommended)

### 6. Configure Custom Domain (Optional)

**Steps**:
1. In Railway, go to your app service
2. Click "Settings" → "Domains"
3. Add your custom domain (e.g., `airabbi.com`)
4. Update DNS records at your registrar:
   - Add CNAME record pointing to Railway's provided domain

**Why**: Professional URL instead of `*.railway.app`

---

### 7. Set Up Error Monitoring (Optional)

**Options**:
- **Sentry** (free tier available): Catches and reports errors
- **LogTail** (free tier): Log aggregation

**Steps for Sentry**:
1. Create account at [sentry.io](https://sentry.io)
2. Create a new Python project
3. Install: `pip install sentry-sdk[flask]`
4. Add to `app.py`:
   ```python
   import sentry_sdk
   sentry_sdk.init(dsn="your-dsn-here")
   ```

---

### 8. Update CLAUDE.md with New Architecture

After migration, update the CLAUDE.md file to reflect:
- pgvector is now the primary database
- New environment variables
- Removed Pinecone keepalive scheduler

---

## Priority: LOW (Nice to Have)

### 9. Remove Pinecone Dependencies (After Successful Migration)

Once pgvector is confirmed working in production:

1. **Update requirements.txt** - remove these lines:
   ```
   pinecone-client==2.2.1
   APScheduler==3.10.4
   ```

2. **Update app.py** - remove Pinecone conditional code

3. **Delete Pinecone index** (optional, to stop any charges):
   - Log into Pinecone console
   - Delete the `midrash` index

---

### 10. Set Up Automated Backups

**Options**:
- Railway provides automatic backups for PostgreSQL
- Configure backup retention in Railway dashboard
- Consider periodic exports for disaster recovery

---

## Verification Checklist

After completing all tasks, verify:

- [ ] Site loads at production URL
- [ ] Example question chips work
- [ ] Search returns results with proper markdown formatting
- [ ] Loading spinner appears during search
- [ ] Rate limit message shows after 5 requests
- [ ] "Copy" button copies response text
- [ ] Mobile layout works correctly
- [ ] Plausible shows visits in dashboard
- [ ] `/health` endpoint returns healthy status
- [ ] Admin logs accessible at `/logs`
- [ ] Error states display correctly (disconnect network and try search)

---

## Troubleshooting

### Migration fails with connection error
- Verify `DATABASE_URL` is correct
- Check Railway service is running
- Try `psql $DATABASE_URL` to test connection

### Search returns no results
- Verify migration completed: `python migrate_to_pgvector.py --verify`
- Check `VECTOR_DB_BACKEND=pgvector` is set
- Check app logs for errors

### Rate limit not working
- Clear cookies and try again
- Check SQLite database exists: `search_logs.db`
- Verify timestamp column format in database

### Plausible not tracking
- Check browser console for blocked scripts
- Verify domain matches exactly
- Wait 5 minutes for data to appear

---

## Support Resources

- **Railway Docs**: https://docs.railway.app/
- **pgvector Docs**: https://github.com/pgvector/pgvector
- **OpenAI API**: https://platform.openai.com/docs/
- **Plausible Docs**: https://plausible.io/docs

---

*Last Updated: January 2026*
