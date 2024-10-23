import java.io.IOException; 
import java.util.HashMap; 
import java.util.Map; 
import java.util.PriorityQueue; 
import java.util.Comparator; 
import java.util.StringTokenizer; 
import org.apache.hadoop.conf.Configuration; 
import org.apache.hadoop.fs.Path; 
import org.apache.hadoop.io.IntWritable; 
import org.apache.hadoop.io.Text; 
import org.apache.hadoop.mapreduce.Job; 
import org.apache.hadoop.mapreduce.Mapper; 
import org.apache.hadoop.mapreduce.Reducer; 
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat; 
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class WordCount {

    public static class TokenizerMapper extends Mapper<Object, Text, Text, IntWritable> {
        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            StringTokenizer itr = new StringTokenizer(value.toString().toLowerCase()); // tokenize and lowercase the text
            while (itr.hasMoreTokens()) {
                String token = itr.nextToken().replaceAll("[^a-zA-Z]", ""); // remove punctuation
                if (token.length() > 0) {
                    word.set(token);
                    context.write(word, one);
                }
            }
        }
    }

    public static class IntSumReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        private IntWritable result = new IntWritable();
        private int totalWords = 0;
        private int uniqueWords = 0;
        private PriorityQueue<Map.Entry<String, Integer>> topWordsQueue;
        private Map<String, Integer> wordCounts = new HashMap<>();

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            topWordsQueue = new PriorityQueue<>(10, Comparator.comparingInt(Map.Entry::getValue));
        }

        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            totalWords += sum;
            uniqueWords++;

            wordCounts.put(key.toString(), sum);
            result.set(sum);
            context.write(key, result);

            topWordsQueue.offer(new HashMap.SimpleEntry<>(key.toString(), sum));
            if (topWordsQueue.size() > 10) {
                topWordsQueue.poll();
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            // Write the summary report
            Path reportPath = new Path("/home/behruzgurbanli/bigdata_as1/output.txt");
            org.apache.hadoop.fs.FileSystem fs = org.apache.hadoop.fs.FileSystem.get(context.getConfiguration());
            try (org.apache.hadoop.fs.FSDataOutputStream out = fs.create(reportPath)) {
                out.writeUTF("\n==== MapReduce Job Summary ====\n");
                out.writeUTF(String.format("%-30s %s\n", "Total Words:", totalWords));
                out.writeUTF(String.format("%-30s %s\n", "Unique Words:", uniqueWords));
                out.writeUTF("\nTop 10 Most Frequent Words:\n");
                
                PriorityQueue<Map.Entry<String, Integer>> reverseQueue = new PriorityQueue<>((a, b) -> Integer.compare(b.getValue(), a.getValue()));
                reverseQueue.addAll(topWordsQueue);

                while (!reverseQueue.isEmpty()) {
                    Map.Entry<String, Integer> entry = reverseQueue.poll();
                    out.writeUTF(String.format("%-15s %s\n", entry.getKey(), entry.getValue()));
                }
                out.writeUTF("=================================\n");
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "word count");
        job.setJarByClass(WordCount.class);
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
