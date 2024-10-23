import java.io.*;
import java.util.*;
import java.util.regex.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class WordCountBigram {

    public static class TokenizerMapper extends Mapper<Object, Text, Text, IntWritable> {
        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();
        private Set<String> stopwords = new HashSet<>();
        private List<String> wordsList = new ArrayList<>();
        private String prevWord = null;

        // Load stopwords from stopword_list.txt
        @Override
        protected void setup(Context context) throws IOException {
            Path stopwordPath = new Path("/home/behruzgurbanli/bigdata_as1/stopword_list.txt");
            FileSystem fs = FileSystem.get(context.getConfiguration());
            BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(stopwordPath)));
            String line;
            while ((line = reader.readLine()) != null) {
                stopwords.add(line.trim().toLowerCase());
            }
            reader.close();
        }

        // Map method that handles word tokenization and bigram generation
        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().toLowerCase();
            StringTokenizer tokenizer = new StringTokenizer(line);

            while (tokenizer.hasMoreTokens()) {
                String token = tokenizer.nextToken().replaceAll("[^a-zA-Z]", "").trim();

                // Skip empty tokens and stopwords
                if (token.isEmpty() || stopwords.contains(token)) {
                    continue;
                }

                // Handle individual word count
                word.set(token);
                context.write(word, one);

                // Generate bigrams
                if (prevWord != null) {
                    String bigram = prevWord + " " + token;
                    word.set(bigram);
                    context.write(word, one);
                }
                prevWord = token;  // Store this token as the previous word for the next iteration
            }
        }
    }

    public static class IntSumReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        private IntWritable result = new IntWritable();
        private Map<String, Integer> wordCounts = new HashMap<>();
        private int totalWords = 0;
        private int uniqueWords = 0;
        private PriorityQueue<Map.Entry<String, Integer>> topWordsQueue;
        private PriorityQueue<Map.Entry<String, Integer>> topBigramsQueue;

        @Override
        protected void setup(Context context) {
            topWordsQueue = new PriorityQueue<>(10, Comparator.comparingInt(Map.Entry::getValue));
            topBigramsQueue = new PriorityQueue<>(10, Comparator.comparingInt(Map.Entry::getValue));
        }

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result); // Emit <word or bigram, count>

            // Differentiate between single words and bigrams
            String keyStr = key.toString();
            if (!keyStr.contains(" ")) {
                totalWords += sum;
                uniqueWords += 1;
                wordCounts.put(keyStr, sum);

                // Update top 10 frequent words
                topWordsQueue.offer(new AbstractMap.SimpleEntry<>(keyStr, sum));
                if (topWordsQueue.size() > 10) {
                    topWordsQueue.poll();
                }
            } else {
                // Handle top bigrams
                topBigramsQueue.offer(new AbstractMap.SimpleEntry<>(keyStr, sum));
                if (topBigramsQueue.size() > 10) {
                    topBigramsQueue.poll();
                }
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException {
            Path reportPath = new Path("/books/output/summary_report.txt");
            FileSystem fs = FileSystem.get(context.getConfiguration());
            try (FSDataOutputStream out = fs.create(reportPath)) {
                out.writeBytes("===== MapReduce Job Summary =====\n");
                out.writeBytes("Total number of words (after stopword removal): " + totalWords + "\n");
                out.writeBytes("Unique words: " + uniqueWords + "\n");

                // Top 10 most frequent words
                out.writeBytes("\nTop 10 Most Frequent Words:\n");
                PriorityQueue<Map.Entry<String, Integer>> reverseWordQueue = new PriorityQueue<>(
                    (e1, e2) -> Integer.compare(e2.getValue(), e1.getValue()));
                reverseWordQueue.addAll(topWordsQueue);
                while (!reverseWordQueue.isEmpty()) {
                    Map.Entry<String, Integer> entry = reverseWordQueue.poll();
                    out.writeBytes(entry.getKey() + ": " + entry.getValue() + "\n");
                }

                // Top 10 most frequent bigrams
                out.writeBytes("\nTop 10 Most Frequent Bigrams:\n");
                PriorityQueue<Map.Entry<String, Integer>> reverseBigramQueue = new PriorityQueue<>(
                    (e1, e2) -> Integer.compare(e2.getValue(), e1.getValue()));
                reverseBigramQueue.addAll(topBigramsQueue);
                while (!reverseBigramQueue.isEmpty()) {
                    Map.Entry<String, Integer> entry = reverseBigramQueue.poll();
                    out.writeBytes(entry.getKey() + ": " + entry.getValue() + "\n");
                }

                out.writeBytes("=================================\n");
            }
        }
    }

    // Main method to configure and run the MapReduce job
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Word Count with Bigrams and Stopwords");
        job.setJarByClass(WordCountBigram.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setCombinerClass(IntSumReducer.class);
        job.setReducerClass(IntSumReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
